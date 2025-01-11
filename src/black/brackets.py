"""Builds on top of nodes.py to track brackets."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final, Optional, Union

from black.nodes import (
    BRACKET,
    CLOSING_BRACKETS,
    COMPARATORS,
    LOGIC_OPERATORS,
    MATH_OPERATORS,
    OPENING_BRACKETS,
    UNPACKING_PARENTS,
    VARARGS_PARENTS,
    is_vararg,
    syms,
)
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node

# types
LN = Union[Leaf, Node]
Depth = int
LeafID = int
NodeType = int
Priority = int


COMPREHENSION_PRIORITY: Final = 20
COMMA_PRIORITY: Final = 18
TERNARY_PRIORITY: Final = 16
LOGIC_PRIORITY: Final = 14
STRING_PRIORITY: Final = 12
COMPARATOR_PRIORITY: Final = 10
MATH_PRIORITIES: Final = {
    token.VBAR: 9,
    token.CIRCUMFLEX: 8,
    token.AMPER: 7,
    token.LEFTSHIFT: 6,
    token.RIGHTSHIFT: 6,
    token.PLUS: 5,
    token.MINUS: 5,
    token.STAR: 4,
    token.SLASH: 4,
    token.DOUBLESLASH: 4,
    token.PERCENT: 4,
    token.AT: 4,
    token.TILDE: 3,
    token.DOUBLESTAR: 2,
}
DOT_PRIORITY: Final = 1


class BracketMatchError(Exception):
    """Raised when an opening bracket is unable to be matched to a closing bracket."""


@dataclass
class BracketTracker:
    """Keeps track of brackets on a line."""

    depth: int = 0
    bracket_match: dict[tuple[Depth, NodeType], Leaf] = field(default_factory=dict)
    delimiters: dict[LeafID, Priority] = field(default_factory=dict)
    previous: Optional[Leaf] = None
    _for_loop_depths: list[int] = field(default_factory=list)
    _lambda_argument_depths: list[int] = field(default_factory=list)
    invisible: list[Leaf] = field(default_factory=list)

    def mark(self, leaf: Leaf) -> None:
        """Mark `leaf` with bracket-related metadata. Keep track of delimiters."""
        if leaf.type == token.COMMENT:
            return

        self.maybe_decrement_after_for_loop_variable(leaf)
        self.maybe_decrement_after_lambda_arguments(leaf)
        bracket_depth_change = 0

        if leaf.type in CLOSING_BRACKETS:
            self.depth -= 1
            if (self.depth, leaf.type) in self.bracket_match:
                opening_bracket = self.bracket_match.pop((self.depth, leaf.type))
                leaf.opening_bracket = opening_bracket
                if not leaf.value:
                    self.invisible.append(leaf)
            leaf.bracket_depth = self.depth
            bracket_depth_change = -1
        
        elif leaf.type in OPENING_BRACKETS:
            self.bracket_match[self.depth, BRACKET[leaf.type]] = leaf
            self.depth += 1
            if not leaf.value:
                self.invisible.append(leaf)
            leaf.bracket_depth = self.depth
            bracket_depth_change = 1

        leaf.bracket_depth = self.depth

        if self.depth == 0:
            self._track_delimiters(leaf)

        if bracket_depth_change == 0:
            self.maybe_increment_lambda_arguments(leaf)
            self.maybe_increment_for_loop_variable(leaf)

        self.previous = leaf

    def any_open_for_or_lambda(self) -> bool:
        """Return True if there is an open for or lambda expression on the line.

        See maybe_increment_for_loop_variable and maybe_increment_lambda_arguments
        for details."""
        return bool(self._for_loop_depths or self._lambda_argument_depths)

    def any_open_brackets(self) -> bool:
        """Return True if there is an yet unmatched open bracket on the line."""
        return bool(self.bracket_match)

    def max_delimiter_priority(self, exclude: Iterable[LeafID] = ()) -> Priority:
        """Return the highest priority of a delimiter found on the line."""
        return max((v for k, v in self.delimiters.items() if k not in exclude), default=0)

    def delimiter_count_with_priority(self, priority: Priority = 0) -> int:
        """Return the number of delimiters with the given `priority`.

        If no `priority` is passed, defaults to max priority on the line.
        """
        if not self.delimiters:
            return 0

        priority = priority or self.max_delimiter_priority()
        return sum(1 for p in self.delimiters.values() if p == priority)

    def maybe_increment_for_loop_variable(self, leaf: Leaf) -> bool:
        """In a for loop, or comprehension, the variables are often unpacks.

        To avoid splitting on the comma in this situation, increase the depth of
        tokens between `for` and `in`.
        """
        if leaf.type == token.NAME and leaf.value == "for":
            self.depth += 1
            self._for_loop_depths.append(self.depth)
            return True

        return False

    def maybe_decrement_after_for_loop_variable(self, leaf: Leaf) -> bool:
        """See `maybe_increment_for_loop_variable` above for explanation."""
        if (
            self._for_loop_depths
            and self._for_loop_depths[-1] == self.depth
            and leaf.type == token.NAME
            and leaf.value == "in"
        ):
            self.depth -= 1
            self._for_loop_depths.pop()
            return True

        return False

    def maybe_increment_lambda_arguments(self, leaf: Leaf) -> bool:
        """In a lambda expression, there might be more than one argument.

        To avoid splitting on the comma in this situation, increase the depth of
        tokens between `lambda` and `:`.
        """
        if leaf.type == token.NAME and leaf.value == "lambda":
            self.depth += 1
            self._lambda_argument_depths.append(self.depth)
            return True

        return False

    def maybe_decrement_after_lambda_arguments(self, leaf: Leaf) -> bool:
        """See `maybe_increment_lambda_arguments` above for explanation."""
        if (
            self._lambda_argument_depths
            and self._lambda_argument_depths[-1] == self.depth
            and leaf.type == token.COLON
        ):
            self.depth -= 1
            self._lambda_argument_depths.pop()
            return True

        return False

    def get_open_lsqb(self) -> Optional[Leaf]:
        """Return the most recent opening square bracket (if any)."""
        return self.bracket_match.get((self.depth - 1, token.RSQB))

    def _track_delimiters(self, leaf: Leaf) -> None:
        """Track delimiters of the leaf at depth 0."""
        delimiter_priority_before = is_split_before_delimiter(leaf, self.previous)
        if delimiter_priority_before and self.previous is not None:
            self.delimiters[id(self.previous)] = delimiter_priority_before
        else:
            delimiter_priority_after = is_split_after_delimiter(leaf)
            if delimiter_priority_after:
                self.delimiters[id(leaf)] = delimiter_priority_after


def is_split_after_delimiter(leaf: Leaf) -> Priority:
    """Return the priority of the `leaf` delimiter, given a line break after it.

    The delimiter priorities returned here are from those delimiters that would
    cause a line break after themselves.

    Higher numbers are higher priority.
    """
    if leaf.type == token.COMMA:
        return COMMA_PRIORITY

    return 0


def is_split_before_delimiter(leaf: Leaf, previous: Optional[Leaf] = None) -> Priority:
    """Return the priority of the `leaf` delimiter, given a line break before it.

    The delimiter priorities returned here are from those delimiters that would
    cause a line break before themselves.

    Higher numbers are higher priority.
    """
    if is_vararg(leaf, within=VARARGS_PARENTS | UNPACKING_PARENTS):
        # * and ** might also be MATH_OPERATORS but in this case they are not.
        # Don't treat them as a delimiter.
        return 0

    if (
        leaf.type == token.DOT
        and leaf.parent
        and leaf.parent.type not in {syms.import_from, syms.dotted_name}
        and (previous is None or previous.type in CLOSING_BRACKETS)
    ):
        return DOT_PRIORITY

    if (
        leaf.type in MATH_OPERATORS
        and leaf.parent
        and leaf.parent.type not in {syms.factor, syms.star_expr}
    ):
        return MATH_PRIORITIES[leaf.type]

    if leaf.type in COMPARATORS:
        return COMPARATOR_PRIORITY

    if (
        leaf.type == token.STRING
        and previous is not None
        and previous.type == token.STRING
    ):
        return STRING_PRIORITY

    if leaf.type not in {token.NAME, token.ASYNC}:
        return 0

    if (
        leaf.value == "for"
        and leaf.parent
        and leaf.parent.type in {syms.comp_for, syms.old_comp_for}
        or leaf.type == token.ASYNC
    ):
        if (
            not isinstance(leaf.prev_sibling, Leaf)
            or leaf.prev_sibling.value != "async"
        ):
            return COMPREHENSION_PRIORITY

    if (
        leaf.value == "if"
        and leaf.parent
        and leaf.parent.type in {syms.comp_if, syms.old_comp_if}
    ):
        return COMPREHENSION_PRIORITY

    if leaf.value in {"if", "else"} and leaf.parent and leaf.parent.type == syms.test:
        return TERNARY_PRIORITY

    if leaf.value == "is":
        return COMPARATOR_PRIORITY

    if (
        leaf.value == "in"
        and leaf.parent
        and leaf.parent.type in {syms.comp_op, syms.comparison}
        and not (
            previous is not None
            and previous.type == token.NAME
            and previous.value == "not"
        )
    ):
        return COMPARATOR_PRIORITY

    if (
        leaf.value == "not"
        and leaf.parent
        and leaf.parent.type == syms.comp_op
        and not (
            previous is not None
            and previous.type == token.NAME
            and previous.value == "is"
        )
    ):
        return COMPARATOR_PRIORITY

    if leaf.value in LOGIC_OPERATORS and leaf.parent:
        return LOGIC_PRIORITY

    return 0


def max_delimiter_priority_in_atom(node: LN) -> Priority:
    """Return maximum delimiter priority inside `node`."""
    if node.type != syms.atom:
        return 0

    first, last = node.children[0], node.children[-1]
    if not (first.type == token.LPAR and last.type == token.RPAR):
        return 0

    bt = BracketTracker()
    for child in node.children[1:-1]:
        if isinstance(child, Leaf):
            bt.mark(child)
        else:
            for leaf in child.leaves():
                bt.mark(leaf)
    
    return bt.max_delimiter_priority(default=0)


def get_leaves_inside_matching_brackets(leaves: Sequence[Leaf]) -> set[LeafID]:
    """Return leaves that are inside matching brackets.

    The input `leaves` can have non-matching brackets at the head or tail parts.
    Matching brackets are included.
    """
    try:
        # Start with the first opening bracket and ignore closing brackets before.
        start_index = next(
            i for i, l in enumerate(leaves) if l.type in OPENING_BRACKETS
        )
    except StopIteration:
        return set()
    bracket_stack = []
    ids = set()
    for i in range(start_index, len(leaves)):
        leaf = leaves[i]
        if leaf.type in OPENING_BRACKETS:
            bracket_stack.append((BRACKET[leaf.type], i))
        if leaf.type in CLOSING_BRACKETS:
            if bracket_stack and leaf.type == bracket_stack[-1][0]:
                _, start = bracket_stack.pop()
                for j in range(start, i + 1):
                    ids.add(id(leaves[j]))
            else:
                break
    return ids
