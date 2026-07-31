import math
import random
import logging
from typing import List, Tuple
import re
import asyncio
import optillm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Node:
    def __init__(self, state: str, action: str, parent: 'Node' = None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children: List[Node] = []
        self.visits = 0
        self.value = 0.0

class RStar:
    def __init__(self, system: str, client, model: str, max_depth: int = 3, num_rollouts: int = 5, c: float = 1.4, request_config: dict = None, request_id: str = None):
        self.client = client
        self.model_name = model
        self.max_depth = max_depth
        self.num_rollouts = num_rollouts
        self.c = c
        self.actions = ["A1", "A2", "A3", "A4", "A5"]
        self.original_question = None
        self.system = system
        self.rstar_completion_tokens = 0
        self.request_id = request_id

        # Extract max_tokens from request_config with default
        self.max_tokens = 4096
        if request_config:
            self.max_tokens = request_config.get('max_tokens', self.max_tokens)

        logger.debug(f"Initialized RStar with model: {model}, max_depth: {max_depth}, num_rollouts: {num_rollouts}")

    async def generate_response_async(self, prompt: str) -> str:
        return await asyncio.to_thread(self.generate_response, prompt)

    async def expand_async(self, node: Node, action: str) -> Node:
        prompt = self.create_prompt(node.state, action)
        new_state = await self.generate_response_async(prompt)
        child_node = Node(new_state, action, node)
        node.children.append(child_node)
        logger.debug(f"Expanded node with action: {action}")
        return child_node

    async def simulate_async(self, node: Node) -> float:
        current_node = node
        depth = 0
        logger.debug("Starting simulation")
        while depth < self.max_depth:
            if not current_node.children:
                action = random.choice(self.actions)
                current_node = await self.expand_async(current_node, action)
            else:
                current_node = random.choice(current_node.children)
            depth += 1
        value = self.evaluate(current_node)
        logger.debug(f"Simulation complete. Final value: {value}")
        return value

    async def mcts_async(self, root_state: str) -> List[Node]:
        root = Node(root_state, None)
        tasks = []
        for _ in range(self.num_rollouts):
            tasks.append(self.mcts_rollout_async(root))
        await asyncio.gather(*tasks)
        return self.extract_trajectories(root)

    async def mcts_rollout_async(self, root: Node):
        node = root
        while node.children:
            node, _ = self.select_action(node)
        action = random.choice(self.actions)
        if len(node.children) < len(self.actions):
            node = await self.expand_async(node, action)
        value = await self.simulate_async(node)
        self.backpropagate(node, value)

    async def solve_async(self, question: str) -> str:
        self.original_question = question
        logger.info(f"Solving question: {question}")
        trajectories = await self.mcts_async(question)
        if not trajectories:
            logger.warning("No trajectories found. Unable to solve the question.")
            return "Unable to solve the question due to insufficient reasoning paths."
        final_trajectory = self.select_final_trajectory(trajectories)
        logger.debug(f"Final trajectory: {[node.state for node in final_trajectory]}")
        answers = [self.extract_answer(node.state) for node in final_trajectory]
        final_answer = self.select_best_answer(answers)
        logger.info(f"Selected final answer: {final_answer}")
        return final_answer, self.rstar_completion_tokens

    def generate_response(self, prompt: str) -> str:
        logger.debug(f"Generating response for prompt: {prompt[:100]}...")
        provider_request = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant focused on solving mathematical problems. Stick to the given question and avoid introducing new scenarios."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.2
        }
        response = self.client.chat.completions.create(**provider_request)
        
        # Log provider call
        if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and self.request_id:
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            optillm.conversation_logger.log_provider_call(self.request_id, provider_request, response_dict)
        
        self.rstar_completion_tokens += response.usage.completion_tokens
        generated_response = response.choices[0].message.content.strip()
        logger.debug(f"Generated response: {generated_response}")
        return generated_response
    
    def select_action(self, node: Node) -> Tuple[Node, str]:
        if not node.children:
            action = random.choice(self.actions)
            logger.debug(f"Selected random action: {action}")
            return node, action

        uct_values = []
        for child in node.children:
            if child.visits == 0:
                uct = float('inf')
            else:
                uct = child.value / child.visits + self.c * math.sqrt(math.log(node.visits) / child.visits)
            uct_values.append(uct)

        best_child = node.children[uct_values.index(max(uct_values))]
        logger.debug(f"Selected action: {best_child.action}")
        return best_child, best_child.action

    def expand(self, node: Node, action: str) -> Node:
        prompt = self.create_prompt(node.state, action)
        new_state = self.generate_response(prompt)
        child_node = Node(new_state, action, node)
        node.children.append(child_node)
        logger.debug(f"Expanded node with action: {action}")
        return child_node

    def simulate(self, node: Node) -> float:
        current_node = node
        depth = 0
        logger.debug("Starting simulation")
        while depth < self.max_depth:
            if not current_node.children:
                action = random.choice(self.actions)
                current_node = self.expand(current_node, action)
            else:
                current_node = random.choice(current_node.children)
            depth += 1
        value = self.evaluate(current_node)
        logger.debug(f"Simulation complete. Final value: {value}")
        return value

    def backpropagate(self, node: Node, value: float):
        logger.debug("Starting backpropagation")
        while node:
            node.visits += 1
            node.value += value
            node = node.parent
        logger.debug("Backpropagation complete")

    def mcts(self, root_state: str) -> List[Node]:
        root = Node(root_state, None)
        logger.debug(f"Starting MCTS with {self.num_rollouts} rollouts")
        for i in range(self.num_rollouts):
            logger.debug(f"Rollout {i+1}/{self.num_rollouts}")
            node = root
            while node.children:
                node, _ = self.select_action(node)
            action = random.choice(self.actions)
            if len(node.children) < len(self.actions):
                node = self.expand(node, action)
            value = self.simulate(node)
            self.backpropagate(node, value)
        logger.debug("MCTS complete")
        return self.extract_trajectories(root)

    def extract_trajectories(self, root: Node) -> List[List[Node]]:
        logger.debug("Extracting trajectories")
        trajectories = []
        stack = [(root, [])]
        while stack:
            node, path = stack.pop()
            if not node.children:
                trajectories.append(path + [node])
            else:
                for child in node.children:
                    stack.append((child, path + [node]))
        logger.debug(f"Extracted {len(trajectories)} trajectories")
        return trajectories

    def mutual_consistency(self, trajectory: List[Node]) -> bool:
        split_index = random.randint(1, len(trajectory) - 1)
        partial_trajectory = trajectory[:split_index]
        prompt = self.create_discriminator_prompt(partial_trajectory)
        completion = self.generate_response(prompt)
        is_consistent = self.compare_completions(completion, trajectory[split_index:])
        logger.debug(f"Mutual consistency check: {'Passed' if is_consistent else 'Failed'}")
        return is_consistent

    def select_final_trajectory(self, trajectories: List[List[Node]]) -> List[Node]:
        logger.debug("Selecting final trajectory")
        valid_trajectories = [t for t in trajectories if self.mutual_consistency(t)]
        logger.debug(f"Found {len(valid_trajectories)} valid trajectories")
        if not valid_trajectories:
            logger.warning("No valid trajectories found. Selecting based on value/visits.")
            return max(trajectories, key=lambda t: self.trajectory_score(t))
        return max(valid_trajectories, key=lambda t: self.trajectory_score(t))

    def trajectory_score(self, trajectory: List[Node]) -> float:
        if not trajectory:
            return float('-inf')
        last_node = trajectory[-1]
        if last_node.visits == 0:
            return last_node.value  # Return just the value if visits is zero
        return last_node.value / last_node.visits

    def select_best_answer(self, answers: List[Tuple[str, float]]) -> str:
        valid_answers = [(answer, conf) for answer, conf in answers if answer]
        if not valid_answers:
            return "Unable to determine a valid answer."
        
        # Sort by confidence and then by frequency
        answer_counts = {}
        for answer, conf in valid_answers:
            if answer in answer_counts:
                answer_counts[answer] = (answer_counts[answer][0] + 1, max(answer_counts[answer][1], conf))
            else:
                answer_counts[answer] = (1, conf)
        
        sorted_answers = sorted(answer_counts.items(), key=lambda x: (-x[1][1], -x[1][0]))
        best_answer, (count, conf) = sorted_answers[0]
        
        logger.debug(f"Selected best answer: {best_answer} (count: {count}, confidence: {conf})")
        return best_answer

    def create_prompt(self, state: str, action: str) -> str:
        question = self.original_question if hasattr(self, 'original_question') else "the original question"
        prompts = {
        "A1": f"""Given the current state: {state}
Generate the next logical step in solving {question}.
Your response should be a single, clear thought that moves towards the solution.
If you can determine the final answer at this step, state it clearly.""",

        "A2": f"""Given the current state: {state}
Continue the reasoning process to solve {question}.
Provide the remaining steps needed to reach the final answer.
Each step should be clear and directly related to solving the problem.""",

        "A3": f"""Given the current state: {state}
Identify a key sub-question that needs to be answered to solve {question}.
State this sub-question clearly, then provide its answer.
Explain how this sub-question and its answer contribute to solving the main problem.""",

        "A4": f"""Given the current state: {state}
Re-examine the previous step in solving {question} using Chain-of-Thought reasoning.
Break down your thinking process explicitly, showing each logical step.
If you reach a conclusion, state it clearly.""",

        "A5": f"""Given the current state: {state}
Rephrase {question} by clearly listing all relevant conditions and unknowns.
Ensure that your rephrasing captures all important details from the original question.
This rephrasing should help clarify the problem and guide the solution process."""
    }
    
        prompt = prompts[action] + "\n\nIf you determine the final answer, explicitly state 'The final answer is [your numeric answer]' at the end of your response."
        logger.debug(f"Created prompt for action {action}: {prompt}")
        return prompt

    def create_discriminator_prompt(self, partial_trajectory: List[Node]) -> str:
        states = [node.state for node in partial_trajectory]
        partial_reasoning = " ".join(states)
        return f"Given the partial reasoning:\n{partial_reasoning}\nComplete the reasoning to solve the problem:"

    def compare_completions(self, completion: str, remaining_trajectory: List[Node]) -> bool:
        remaining_states = [node.state for node in remaining_trajectory]
        remaining_reasoning = " ".join(remaining_states)
        
        # Normalize both strings: remove punctuation, convert to lowercase, and split into words
        completion_words = set(completion.lower().replace('.', '').replace(',', '').split())
        trajectory_words = set(remaining_reasoning.lower().replace('.', '').replace(',', '').split())
        
        # Calculate word overlap
        overlap = len(completion_words.intersection(trajectory_words))
        total_words = len(completion_words.union(trajectory_words))
        
        # Consider it a match if there's more than 70% word overlap
        return overlap / total_words > 0.7

    def evaluate(self, node: Node) -> float:
        # Extract the final answer from the node's state
        answer, confidence = self.extract_answer(node.state)
        
        # Check if the answer is a number
        try:
            float(answer)
            logger.debug(f"Evaluated node. Answer: {answer}, Confidence: {confidence}, Value: {confidence}")
            return confidence  # Return the confidence as the value
        except ValueError:
            logger.debug(f"Evaluated node. Answer: {answer}, Confidence: {confidence}, Value: 0.0")
            return 0.0  # If it's not a valid number, return a low score

    def extract_answer(self, final_state: str) -> Tuple[str, float]:
        logger.debug(f"Extracting answer from state: {final_state}")
        patterns = [
            r"The answer is (\d+)",
            r"The final answer is (\d+)",
            r"Therefore, the answer is (\d+)",
            r"So, the answer is (\d+)",
            r"Thus, the answer is (\d+)",
            r"In conclusion, the answer is (\d+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, final_state)
            if match:
                answer = match.group(1)
                confidence = 1.0
                logger.debug(f"Answer found using pattern '{pattern}': {answer}")
                return answer, confidence
        
        # If no pattern is found, try to extract any number
        numbers = re.findall(r'\d+', final_state)
        if numbers:
            answer = numbers[-1]  # Take the last number found
            confidence = 0.5  # Lower confidence as it's not in the expected format
            logger.debug(f"No pattern found. Using last number as answer: {answer}")
            return answer, confidence
        
        logger.warning("No answer found in the state.")
        return "", 0.0
   
    def solve(self, question: str) -> str:
        """
        Synchronous wrapper for solve_async method.
        """
        return asyncio.run(self.solve_async(question))