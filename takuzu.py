# takuzu.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 00000 Nome1
# 00000 Nome2

import sys
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)

from utils import *

ZERO = 0
ONE = 1
EMPTY = 2
IMPOSSIBLE = -1


class TakuzuState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id


class Board:
    """Representação interna de um tabuleiro de Takuzu."""

    def __init__(self, size, board):
        self.size = size
        self.board = board

        self.emptyPositions = []

        self.rows = []
        self.columns = []

        self.horizontalBinaryValues = []
        self.verticalBinaryValues = []

    def __str__(self):
        output = ""
        # for line in self.board:
        #    output += ("\t".join(line)+"\n")
        for y in range(self.size):
            for x in range(self.size-1):
                output += str(Board.get_number(self, x, y))+"\t"
            output += str(Board.get_number(self, self.size-1, y))
            if y != self.size-1:
                output += "\n"
        return output

    class BoardLine:  # Columns and Rows of a Board
        def __init__(self, maxCount, counterZero, counterOne):
            self.maxCount = maxCount
            # How many zeros there are in the line
            self.counter = [counterZero, counterOne]

        # Returns the number that must be played on the line. 2 if none is mandatory
        def getObligatoryPlay(self):
            # If it's more than half full of something, then the Board is already wrong
            for count in (self.counter[ZERO], self.counter[ONE]):
                if count > self.maxCount:
                    return IMPOSSIBLE

            # If it's half full of Zeros, must return 1
            if self.counter[0] == self.maxCount:
                return ONE

            # If it's half full of Ones, must return 0
            if self.counter[1] == self.maxCount:
                return ZERO

            return None  # Neither limit is reached and there is no obligatory play

#    def calculateBinaryNumber(numberRow):

    def calculateAuxiliaryStats(self):
        # Initiate all of the board lines with the correct number of zeros and ones

        for i in range(self.size):  # Columns
            self.columns += [self.BoardLine(
                np.ceil(self.size / 2), self.board[i].count(ZERO), self.board[i].count(ONE))]

        # Binary value calculation for all columns
        for i in range(0, self.size):
            skipCurrentLine = False
            currentBinaryNumber = 0

            for j in range(0, self.size):
                if self.board[i][j] == 2:  # Line wasn't full, can be skipped
                    skipCurrentLine = True
                    break
                else:
                    currentBinaryNumber += self.board[i][j] * \
                        (self.size - 1 - j)

            if not skipCurrentLine:  # If line wasn't full, then add the binary number to the board
                self.verticalBinaryValues += [currentBinaryNumber]

        for i in range(self.size):  # Rows
            # Because the row is divided into the columns' arrays, they must be converted into a single list
            rowToList = []
            for j in range(self.size):
                rowToList += [self.board[j][i]]

            self.rows += [self.BoardLine(
                np.ceil(self.size / 2), rowToList.count(ZERO), rowToList.count(ONE))]

        # Binary value calculation for all rows
        for j in range(0, self.size):
            skipCurrentLine = False
            currentBinaryNumber = 0

            for i in range(0, self.size):
                if self.board[i][j] == 2:  # Line wasn't full, can be skipped
                    skipCurrentLine = True
                    break
                else:
                    currentBinaryNumber += self.board[i][j] * \
                        (self.size - 1 - j)

            if not skipCurrentLine:  # If line wasn't full, then add the binary number to the board]
                self.horizontalBinaryValues += [currentBinaryNumber]

    def get_number(self, x: int, y: int) -> int:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[x][y]

    def adjacent_vertical_numbers(self, x: int, y: int) -> (int, int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        below, above = None, None
        if y < self.size - 1:
            below = Board.get_number(self, x, y+1)
        if y > 0:
            above = Board.get_number(self, x, y-1)
        return (below, above)

    def adjacent_vertical_numbers_expanded(self, x: int, y: int) -> (int, int, int, int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        belowFar, below, above, aboveFar = None, None, None, None
        if y < self.size - 1:
            below = Board.get_number(self, x, y+1)
        if y < self.size - 2:
            belowFar = Board.get_number(self, x, y+2)

        if y > 0:
            above = Board.get_number(self, x, y-1)
        if y > 1:
            aboveFar = Board.get_number(self, x, y-2)

        return (belowFar, below, above, aboveFar)

    def adjacent_horizontal_numbers(self, x: int, y: int) -> (int, int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        left, right = None, None
        if x < self.size - 1:
            right = Board.get_number(self, x+1, y)
        if x > 0:
            left = Board.get_number(self, x-1, y)
        return (left, right)

    def adjacent_horizontal_numbers_expanded(self, x: int, y: int) -> (int, int, int, int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        leftFar, left, right, rightFar = None, None, None, None
        if x < self.size - 1:
            right = Board.get_number(self, x+1, y)
        if x < self.size - 2:
            rightFar = Board.get_number(self, x+2, y)

        if x > 0:
            left = Board.get_number(self, x-1, y)
        if x > 1:
            leftFar = Board.get_number(self, x-2, y)

        return (leftFar, left, right, rightFar)

    def get_emptyPositions(self):
        positions = []
        for x in range(self.size):
            for y in range(self.size):
                if self.get_number(x, y) == EMPTY:
                    positions += [(x, y)]
        return positions

    def copy_Board(self):
        newBoard = [[] for _ in range(self.size)]
        for x in range(self.size):
            for y in range(self.size):
                newBoard[x] += [self.get_number(x, y)]
        return newBoard

    def play(self, action):
        """Coloca um número numa posição do tabuleiro."""
        temp = self.copy_Board()

        temp[action.x][action.y] = action.number

        emptyPositions = []
        for position in self.emptyPositions:
            if position != (action.x, action.y):
                emptyPositions += [(position[0], position[1])]

        newBoard = Board(self.size, temp)
        newBoard.emptyPositions = emptyPositions

        newRows, newColumns = [], []
        for row in range(self.size):
            if row != action.y:
                newRows += [newBoard.BoardLine(np.ceil(newBoard.size / 2),
                                               self.rows[row].counter[ZERO], self.rows[row].counter[ONE])]
            else:
                if action.number == ZERO:
                    newRows += [newBoard.BoardLine(np.ceil(newBoard.size / 2),
                                                   self.rows[row].counter[ZERO] + 1, self.rows[row].counter[ONE])]
                else:
                    newRows += [newBoard.BoardLine(np.ceil(newBoard.size / 2),
                                                   self.rows[row].counter[ZERO], self.rows[row].counter[ONE]+1)]

        for column in range(self.size):
            if column != action.x:
                newColumns += [newBoard.BoardLine(np.ceil(newBoard.size / 2),
                                                  self.columns[column].counter[ZERO], self.columns[column].counter[ONE])]
            else:
                if action.number == ZERO:
                    newColumns += [newBoard.BoardLine(np.ceil(newBoard.size / 2),
                                                      self.columns[column].counter[ZERO] + 1, self.columns[column].counter[ONE])]
                else:
                    newColumns += [newBoard.BoardLine(np.ceil(newBoard.size / 2),
                                                      self.columns[column].counter[ZERO], self.columns[column].counter[ONE] + 1)]

        newBoard.rows = newRows
        newBoard.columns = newColumns

        newBoard.horizontalBinaryValues = self.horizontalBinaryValues
        if action.horizontalBinaryValue != None:
            newBoard.horizontalBinaryValues += [action.horizontalBinaryValue]

        newBoard.verticalBinaryValues = self.verticalBinaryValues
        if action.verticalBinaryValue != None:
            newBoard.verticalBinaryValues += [action.verticalBinaryValue]

        return newBoard

    @staticmethod
    def parse_instance_from_stdin():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 takuzu.py < input_T01

            > from sys import stdin
            > stdin.readline()
        """
        size = int(sys.stdin.readline().split('\n')[0])
        board = [[] for _ in range(size)]
        for _ in range(size):
            line = sys.stdin.readline().replace('\n', '\t').split(
                '\t')[:-1]  # Parse the line into a list of separate strings
            for i in range(size):
                board[i] += [int(line[i])]
        newBoard = Board(size, board)
        newBoard.emptyPositions = newBoard.get_emptyPositions()
        newBoard.calculateAuxiliaryStats()

        return newBoard


"""

"""


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = TakuzuState(board)

    class Action:
        def __init__(self, x: int, y: int, number):
            self.x = x  # Starts at 1
            self.y = y  # Starts at 1
            self.number = number  # 2 is empty
            self.horizontalBinaryValue = None
            self.verticalBinaryValue = None

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento.

            If there is an obligatory action, only that action will be returned. Else,
        all possible actions are returned.

            The first actions to be analyzed are the ones derived from the 50/50 rule.

            The second ones are seen position by position.

            If no obligatory one is found, a big list  of all possible actions is 
            generated as said above.
        """
        # Looks through all actions given to it and returns all that don't break any rules
        def finishActionLookup(actions):
            actionsResult = []

            # If an impossible obligatory action was found, then return a null list
            if actions[0].number == IMPOSSIBLE:
                return []

            # Check if an action will complete a row or a column. If it will, and the row or column is repeated, then rule out the action
            for action in actions:
                # Whether or not the action should be added
                addAction = True

                # The summed values of the row and column that the action is being inserted in
                horizontalBinaryValue = 0
                verticalBinaryValue = 0

                count = 0
                # Check the row
                for x in reversed(range(0, state.board.size)):
                    if state.board.get_number(x, action.y) == EMPTY and x != action.x:
                        horizontalBinaryValue = None
                        break

                    if x == action.x:
                        horizontalBinaryValue += action.number * (2 ** count)
                    else:
                        horizontalBinaryValue += state.board.get_number(x,action.y) * (2 ** count)
                    count += 1
                
                if horizontalBinaryValue != None and horizontalBinaryValue in state.board.horizontalBinaryValues:
                    addAction = False
                else:
                    action.horizontalBinaryValue = horizontalBinaryValue
                
                count = 0
                # Check the column
                if addAction:
                    for y in reversed(range(0, state.board.size)):
                        if state.board.get_number(action.x, y) == EMPTY and y != action.y:
                            verticalBinaryValue = None
                            break
                        
                        if y == action.y:
                            verticalBinaryValue += action.number * (2 ** count)
                        else:
                            verticalBinaryValue += state.board.get_number(action.x,y) * (2 ** count)
                        count += 1

                    if verticalBinaryValue != None and verticalBinaryValue in state.board.verticalBinaryValues:
                        addAction = False
                    else:
                        action.verticalBinaryValue = verticalBinaryValue

                # The action does not break any rules and will, as such, be added to the list that gets returned
                if addAction:
                    actionsResult += [action]

            return actionsResult

        # Get obligatory play derived from the 50/50 rule
        for i in range(0, state.board.size):  # Row by row. i is row and j is column
            num = state.board.rows[i].getObligatoryPlay()

            if num == IMPOSSIBLE:
                return []
            if num != None:  # If an obligatory play is found
                for j in range(state.board.size):  # Find an empty spot in the row
                    if state.board.get_number(j, i) == EMPTY:
                        # print("Play Found in #1")
                        # And return the correct action
                        return finishActionLookup([self.Action(j, i, num)])

        # Column by column. i is column and j is row
        for i in range(state.board.size):
            num = state.board.columns[i].getObligatoryPlay()

            if num == IMPOSSIBLE:
                return []
            if num != None:  # If an obligatory play is found
                for j in range(state.board.size):  # Find an empty spot in the column
                    if state.board.get_number(i, j) == EMPTY:
                        # print("Play Found in #2")
                        # And return the correct action
                        return finishActionLookup([self.Action(i, j, num)])

        # Returns the obligatory play for an empty position, considering the nearest 4 positions
        def playDict(numberRow):
            opposite = (ONE, ZERO)

            left, middle, right = EMPTY, EMPTY, EMPTY
            if numberRow[0] == numberRow[1]:
                left = numberRow[1]
            if numberRow[3] == numberRow[2]:
                right = numberRow[2]
            if numberRow[1] == numberRow[2]:
                middle = numberRow[1]

            # Left, Right and Middle should be understood as: XX__ = LEFT, __XX = RIGHT, _XX_ = MIDDLE
            if left in opposite:
                if right in opposite:
                    if left == right:
                        return opposite[left]  # 11[]11 -> 11[0]11
                    return IMPOSSIBLE   # 11[error]00
                return opposite[left]   # 11[0]__ / etc
            if right in opposite:
                return opposite[right]  # __[1]00 / etc
            if middle in opposite:
                return opposite[middle]  # _1[0]1_
            return None

        # Returns an obligatory number to play for a specific position if there is one, else returns None
        def getObligatoryPlayForPosition(x, y):
            # Horizontal
            horizontalNums = state.board.adjacent_horizontal_numbers_expanded(
                x, y)

            correctPlay = playDict(horizontalNums)
            if correctPlay != None:  # If there is a play that should be made, return it
                return correctPlay

            # Vertical
            verticalNums = state.board.adjacent_vertical_numbers_expanded(x, y)

            correctPlay = playDict(verticalNums)
            if correctPlay != None:  # If there is a play that should be made, return it
                return correctPlay

            return None

        # Look in all empty positions for a obligatory play to make
        for position in state.board.emptyPositions:
            obligatoryPlay = getObligatoryPlayForPosition(
                position[0], position[1])

            if obligatoryPlay == IMPOSSIBLE:
                return []
            if obligatoryPlay != None:
                # print("Play Found in #3")
                return finishActionLookup([self.Action(position[0],
                                                       position[1], obligatoryPlay)])

        # Since no obligatory play was found, return two actions for the first empty position

        position = state.board.emptyPositions[0]
        # print("Play Found in #4")
        return finishActionLookup([self.Action(position[0], position[1], 0),
                                   self.Action(position[0], position[1], 1)])

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        newBoard = state.board.play(action)

        # print("Played: (" + str(action.x) + "," + str(action.y) + ")\n", newBoard, sep="")

        return TakuzuState(newBoard)

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        for row in range(state.board.size):
            for col in range(state.board.size):
                if state.board.get_number(row, col) == EMPTY:
                    return False
        return True

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    # DOING:
    # Ler o ficheiro do standard input, OK
    # Usar uma técnica de procura para resolver a instância, SOB
    # Retirar a solução a partir do nó resultante, SOB
    # Imprimir para o standard output no formato indicado. OK
    board = Board.parse_instance_from_stdin()
    # print("Initial:\n", board, sep="")

    problem = Takuzu(board)

    goal_node = depth_first_tree_search(problem)

    # print("Is goal?", problem.goal_test(goal_node.state))
    print(goal_node.state.board, sep="")
