import random
import logging
import numpy as np
import networkx as nx
from typing import List, Dict
from optillm import conversation_logger

logger = logging.getLogger(__name__)

class DialogueState:
    def __init__(self, system_prompt: str, conversation_history: List[Dict[str, str]], current_query: str):
        self.system_prompt = system_prompt
        self.conversation_history = conversation_history
        self.current_query = current_query

    def __str__(self):
        return f"System: {self.system_prompt}\nHistory: {self.conversation_history}\nCurrent Query: {self.current_query}"

class MCTSNode:
    def __init__(self, state: DialogueState, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

class MCTS:
    def __init__(self, simulation_depth, exploration_weight, client, model, request_config=None, request_id=None):
        self.simulation_depth = simulation_depth
        self.exploration_weight = exploration_weight
        self.root = None
        self.graph = nx.Graph()
        self.node_labels = {}
        self.client = client
        self.model = model
        self.completion_tokens = 0
        self.request_id = request_id

        # Extract max_tokens from request_config with default
        self.max_tokens = 4096
        if request_config:
            self.max_tokens = request_config.get('max_tokens', self.max_tokens)

    def select(self, node: MCTSNode) -> MCTSNode:
        logger.debug(f"Selecting node. Current node visits: {node.visits}, value: {node.value}")
        if not node.children:
            logger.debug("Node has no children. Returning current node.")
            return node
        selected_node = max(node.children, key=lambda c: c.value / (c.visits + 1e-8) + self.exploration_weight * np.sqrt(np.log(node.visits + 1) / (c.visits + 1e-8)))
        logger.debug(f"Selected child node. Visits: {selected_node.visits}, Value: {selected_node.value}")
        return selected_node

    def expand(self, node: MCTSNode) -> MCTSNode:
        logger.debug(f"Expanding node. Current state: {node.state}")
        actions = self.generate_actions(node.state)
        logger.debug(f"Generated {len(actions)} possible actions")
        for i, action in enumerate(actions):
            new_state = self.apply_action(node.state, action)
            child = MCTSNode(new_state, parent=node)
            node.children.append(child)
            self.graph.add_edge(id(node), id(child))
            self.node_labels[id(child)] = f"Visits: {child.visits}\nValue: {child.value:.2f}"
            logger.debug(f"Created child node {i+1}. Action: {action[:50]}...")
        selected_child = random.choice(node.children)
        logger.debug(f"Randomly selected child node for simulation. Visits: {selected_child.visits}, Value: {selected_child.value}")
        return selected_child

    def simulate(self, node: MCTSNode) -> float:
        logger.debug(f"Starting simulation from node. Current query: {node.state.current_query}")
        state = node.state
        for i in range(self.simulation_depth):
            if self.is_terminal(state):
                logger.debug(f"Reached terminal state at depth {i}")
                break
            action = random.choice(self.generate_actions(state))
            state = self.apply_action(state, action)
            logger.debug(f"Simulation step {i+1}. Action: {action[:50]}...")
        value = self.evaluate_state(state)
        logger.debug(f"Simulation complete. Final state value: {value}")
        return value

    def backpropagate(self, node: MCTSNode, value: float):
        logger.debug(f"Starting backpropagation. Initial value: {value}")
        while node:
            node.visits += 1
            node.value += value
            self.node_labels[id(node)] = f"Visits: {node.visits}\nValue: {node.value:.2f}"
            logger.debug(f"Updated node. Visits: {node.visits}, New value: {node.value}")
            node = node.parent

    def search(self, initial_state: DialogueState, num_simulations: int) -> DialogueState:
        logger.debug(f"Starting MCTS search with {num_simulations} simulations")
        if not self.root:
            self.root = MCTSNode(initial_state)
            self.graph.add_node(id(self.root))
            self.node_labels[id(self.root)] = "Root\nVisits: 0\nValue: 0.00"
            logger.debug("Created root node")
        
        for i in range(num_simulations):
            logger.debug(f"Starting simulation {i+1}")
            node = self.select(self.root)
            if not self.is_terminal(node.state):
                node = self.expand(node)
            value = self.simulate(node)
            self.backpropagate(node, value)
            
        best_child = max(self.root.children, key=lambda c: c.visits)
        logger.debug(f"Search complete. Best child node: Visits: {best_child.visits}, Value: {best_child.value}")
        return best_child.state

    def generate_actions(self, state: DialogueState) -> List[str]:
        logger.debug("Generating actions for current state")
        messages = [{"role": "system", "content": state.system_prompt}]
        messages.extend(state.conversation_history)
        messages.append({"role": "user", "content": state.current_query})
        
        completions = []
        n = 3

        logger.info(f"Requesting {n} completions from the model")
        provider_request = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "n": n,
            "temperature": 1
        }
        response = self.client.chat.completions.create(**provider_request)

        # Log provider call
        if self.request_id:
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            conversation_logger.log_provider_call(self.request_id, provider_request, response_dict)

        # Check for valid response with None-checking
        if response is None or not response.choices:
            logger.error("Failed to get valid completions from the model")
            return []

        completions = [choice.message.content.strip() for choice in response.choices if choice.message.content is not None]
        self.completion_tokens += response.usage.completion_tokens
        logger.info(f"Received {len(completions)} completions from the model")
        return completions

    def apply_action(self, state: DialogueState, action: str) -> DialogueState:
        logger.info(f"Applying action: {action[:50]}...")
        new_history = state.conversation_history.copy()
        new_history.append({"role": "assistant", "content": action})
        
        messages = [{"role": "system", "content": state.system_prompt}]
        messages.extend(new_history)
        messages.append({"role": "user", "content": "Based on this conversation, what might the user ask or say next? Provide a likely user query."})
        
        logger.info("Requesting next user query from the model")
        provider_request = {
            "model": self.model,
            "messages": messages,
            "max_tokens": min(self.max_tokens, 1024),
            "n": 1,
            "temperature": 1
        }
        response = self.client.chat.completions.create(**provider_request)

        # Log provider call
        if self.request_id:
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            conversation_logger.log_provider_call(self.request_id, provider_request, response_dict)

        # Check for valid response with None-checking
        if (response is None or
            not response.choices or
            response.choices[0].message.content is None or
            response.choices[0].finish_reason == "length"):
            logger.warning("Next query response truncated or empty, using default")
            next_query = "Please continue."
        else:
            next_query = response.choices[0].message.content

        self.completion_tokens += response.usage.completion_tokens
        logger.info(f"Generated next user query: {next_query}")
        return DialogueState(state.system_prompt, new_history, next_query)

    def is_terminal(self, state: DialogueState) -> bool:
        is_terminal = len(state.conversation_history) > 10 or "goodbye" in state.current_query.lower()
        logger.info(f"Checking if state is terminal: {is_terminal}")
        return is_terminal

    def evaluate_state(self, state: DialogueState) -> float:
        logger.info("Evaluating current state")
        messages = [{"role": "system", "content": state.system_prompt}]
        messages.extend(state.conversation_history)
        messages.append({"role": "user", "content": "Evaluate the quality of this conversation on a scale from 0 to 1, where 0 is poor and 1 is excellent. Consider factors such as coherence, relevance, and engagement. Respond with only a number."})
        
        provider_request = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 256,
            "n": 1,
            "temperature": 0.1
        }
        response = self.client.chat.completions.create(**provider_request)

        # Log provider call
        if self.request_id:
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            conversation_logger.log_provider_call(self.request_id, provider_request, response_dict)

        self.completion_tokens += response.usage.completion_tokens

        # Check for valid response with None-checking
        if (response is None or
            not response.choices or
            response.choices[0].message.content is None or
            response.choices[0].finish_reason == "length"):
            logger.warning("Evaluation response truncated or empty. Using default value 0.5")
            return 0.5

        try:
            score = float(response.choices[0].message.content.strip())
            score = max(0, min(score, 1))  # Ensure the score is between 0 and 1
            logger.info(f"State evaluation score: {score}")
            return score
        except ValueError:
            logger.warning("Failed to parse evaluation score. Using default value 0.5")
            return 0.5  # Default to a neutral score if parsing fails

def chat_with_mcts(system_prompt: str, initial_query: str, client, model: str, num_simulations: int = 2, exploration_weight: float = 0.2,
                   simulation_depth: int = 1, request_config: dict = None, request_id: str = None) -> str:
    logger.info("Starting chat with MCTS")
    logger.info(f"Parameters: num_simulations={num_simulations}, exploration_weight={exploration_weight}, simulation_depth={simulation_depth}")
    mcts = MCTS(simulation_depth=simulation_depth, exploration_weight=exploration_weight, client=client, model=model, request_config=request_config, request_id=request_id)
    initial_state = DialogueState(system_prompt, [], initial_query)
    logger.info(f"Initial query: {initial_query}")
    final_state = mcts.search(initial_state, num_simulations)
    response = final_state.conversation_history[-1]['content'] if final_state.conversation_history else ""
    logger.info(f"MCTS chat complete. Final response: {response[:100]}...")
    return response, mcts.completion_tokens