"""Offline regression tests for selection through the MCTS search tree."""

from unittest.mock import Mock

from optillm.mcts import MCTS, DialogueState, MCTSNode


def make_node(query, parent=None, visits=1, value=0.0):
    node = MCTSNode(DialogueState("system", [], query), parent=parent)
    node.visits = visits
    node.value = value
    if parent is not None:
        parent.children.append(node)
    return node


def test_selection_follows_uct_at_each_depth():
    mcts = MCTS(0, 0.2, None, "test-model")
    root = make_node("root", visits=10)
    preferred = make_node("preferred", root, visits=5, value=4.0)
    make_node("other", root, visits=5, value=1.0)
    make_node("visited", preferred, visits=5, value=4.0)
    unexplored = make_node("unexplored", preferred, visits=0)

    assert mcts.select(root) is unexplored


def test_selection_returns_leaf_unchanged():
    mcts = MCTS(0, 0.2, None, "test-model")
    leaf = make_node("leaf")

    assert mcts.select(leaf) is leaf


def test_search_expands_each_selected_leaf_once(monkeypatch):
    mcts = MCTS(0, 0.2, None, "test-model")
    initial_state = DialogueState("system", [], "question")
    generated_from = []

    def generate_actions(state):
        generated_from.append(state.current_query)
        return [state.current_query + " -> next"]

    monkeypatch.setattr(mcts, "generate_actions", generate_actions)
    monkeypatch.setattr(
        mcts, "apply_action", lambda state, action: DialogueState("system", [], action)
    )
    monkeypatch.setattr(mcts, "evaluate_state", Mock(return_value=0.5))

    mcts.search(initial_state, num_simulations=3)

    assert generated_from == ["question", "question -> next", "question -> next -> next"]
    assert mcts.root.visits == 3
    assert len(mcts.root.children[0].children) == 1
