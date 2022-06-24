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

    class BoardLine:  # Columns and Rows of a Board
        def __init__(self, maximumCount, counterZero, counterOne):
            self.maxCount = maximumCount
            # How many zeros there are in the line
            self.counter[0] = counterZero
            self.counter[1] = counterOne  # How many ones there are in the line

        # Adds either a zero or a one to the number counter
        def incrementCounter(self, num):
            self.counter[num] += 1

        # Returns the number that must be played on the line. 2 if none is mandatory
        def getObligatoryPlay(self):
            # If they're equal, no obligatory play
            if self.counter[0] == self.counter[1]:
                return None

            # If it's half full of Zeros, must return 1
            if self.counter[0] == self.maxCount:
                return 1

            # If it's half full of Ones, must return 0
            if self.counter[1] == self.maxCount:
                return 0

            return None  # Neither limit is reached and there is no obligatory play

    def __init__(self, size, board):
        self.size = size
        self.board = board
        
        self.emptyPositions = []

        self.rows = []
        self.columns = []

        self.horizontalBinaryValues = []
        self.verticalBinaryValues = []

    def calculateAuxiliaryStats(self):
        # Initiate all of the board lines with the correct number of zeros and ones

        for i in range(self.size):  # Rows
            self.rows[i] = self.BoardLine(
                self.size / 2, self.board[i].count[0], self.board[i].count[1])

        # Binary value calculation for all rows
        for i in range(0, self.size):
            skipCurrentLine = False
            currentBinaryNumber = 0

            for j in range(0, self.size):
                if self.board[i][j] == 2:  # Line wasn't full, can be skipped
                    skipCurrentLine = True
                    break
                else:
                    currentBinaryNumber += self.board[i][j] * \
                        (self.board.size - 1 - j)

            if not skipCurrentLine:  # If line wasn't full, then add the binary number to the board
                self.horizontalBinaryValues += [currentBinaryNumber]

        for i in range(self.size):  # Columns
            # Because the column is divided into the rows' arrays, they must be converted into a single list
            columnToList = []
            for j in range(self.size):
                columnToList += [self.board[j][i]]

            self.columns[i] = self.BoardLine(
                self.size / 2, columnToList.count(0), columnToList.count[1])

        # Binary value calculation for all columns
        for j in range(0, self.size):
            skipCurrentLine = False
            currentBinaryNumber = 0

            for i in range(0, self.size):
                if self.board[i][j] == 2:  # Line wasn't full, can be skipped
                    skipCurrentLine = True
                    break
                else:
                    currentBinaryNumber += self.board[i][j] * \
                        (self.board.size - 1 - j)

            if not skipCurrentLine:  # If line wasn't full, then add the binary number to the board
                self.verticalBinaryValues += [currentBinaryNumber]

    def __str__(self):
        output = ""
        # for line in self.board:
        #    output += ("\t".join(line)+"\n")
        for row in range(self.size):
            for col in range(self.size-1):
                output += str(Board.get_number(self, row, col))+"\t"
            output += str(Board.get_number(self, row, self.size))+"\n"
        return output

    def get_number(self, row: int, col: int) -> int:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[row][col]

    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        below, above = None, None
        if row < self.size:
            below = Board.get_number(self, row+1, col)
        if row > 0:
            above = Board.get_number(self, row-1, col)
        return (below, above)

    def adjacent_vertical_numbers_expanded(self, row: int, col: int) -> (int, int, int, int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        belowFar, below, above, aboveFar = None, None, None, None
        if row < self.size:
            below = Board.get_number(self, row+1, col)
        if row < self.size + 1:
            belowFar = Board.get_number(self, row+2, col)

        if row > 0:
            above = Board.get_number(self, row-1, col)
        if row > 1:
            aboveFar = Board.get_number(self, row-2, col)

        return (belowFar, below, above, aboveFar)

    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        left, right = None, None
        if col < self.size:
            right = Board.get_number(self, row, col+1)
        if col > 0:
            left = Board.get_number(self, row, col-1)
        return (left, right)

    def adjacent_horizontal_numbers_expanded(self, row: int, col: int) -> (int, int, int, int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        leftFar, left, right, rightFar = None, None, None, None
        if col < self.size:
            right = Board.get_number(self, row, col+1)
        if col < self.size + 1:
            rightFar = Board.get_number(self, row, col+2)

        if col > 0:
            left = Board.get_number(self, row, col-1)
        if col > 1:
            leftFar = Board.get_number(self, row, col-2)

        return (leftFar, left, right, rightFar)

    def get_emptyPositions(self):
        positions = []
        for row in range(self.size):
            for col in range(self.size):
                if self.get_number(row,col) == 2:
                    positions += [(row, col)]
        return positions

    def copy_Board(self):
        newBoard = []
        for row in range(self.size):
            line = []
            for col in range(self.size):
                line += [self.get_number(row,col)]
            newBoard += [line]
        return newBoard

    def play(self, action):
        """Coloca um número numa posição do tabuleiro."""
        temp = self.copy_Board()

        row, col, number = action[0][0], action[0][1], action[1]
        temp[row][col] = number

        emptyPositions = []
        for position in self.emptyPositions:
            if position != action[0]:
                emptyPositions += [action[0]]

        newBoard = Board(self.size, temp)
        newBoard.emptyPositions = emptyPositions

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
        board, emptyPositions = [], []
        size = int(sys.stdin.readline().split('\n')[0])
        for r in range(size):
            line = sys.stdin.readline().replace('\n', '\t').split('\t')[:-1]  # Parse the line into a list of separate strings
            line = [int(x) for x in line]
            board += [line]
        newBoard = Board(size, board)
        newBoard.emptyPositions = newBoard.get_emptyPositions()
        newBoard.calculateAuxiliaryStats()

        return newBoard

    # TODO: outros metodos da classe


"""

"""


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        # TODO
        pass

    class Action:
        def __init__(self, row: int, col: int, number):
            self.row = row  # Starts at 1
            self.col = col  # Starts at 1
            self.number = number  # 2 is empty

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
            if actions[0].number == -1:
                return []

            for action in actions:  # Runs through each action

                # Horizontal Checking
                # Calculates number that will be added to bitmasking
                horizontalBinaryValue = action.number * \
                    (self.board.size - 1 - action.col)

                # Calculate binary values of each completed row and column

            pass

            # Get obligatory play derived from the 50/50 rule
        for i in range(0, self.board.size):  # Row by row. i is row and j is column
            num = self.board.rows[i].getObligatoryPlay()

            if num != None:  # If an obligatory play is found
                for j in range(0, self.board.size):  # Find an empty spot in the row
                    if self.board.get_number(i, j) == 2:
                        # And return the correct action
                        return finishActionLookup([self.Action(i, j, num)])

        # Column by column. i is column and j is row
        for i in range(0, self.board.size):
            num = self.board.columns[i].getObligatoryPlay()

            if num != None:  # If an obligatory play is found
                for j in range(0, self.board.size):  # Find an empty spot in the column
                    if self.board.get_number(j, i) == 2:
                        # And return the correct action
                        return finishActionLookup([self.Action(j, i, num)])

        # Returns an obligatory number to play for a specific position if there is one, else returns None
        def getObligatoryPlayForPosition(row, column):  # Starts at 1
            playDict = {
                "0011": -1, "1100": -1, "1101": -1, "0010": -1,
                "0022": 1, "2200": 1, "2002": 1,
                "1122": 0, "2211": 0, "2112": 0
            }

            # Horizontal
            horizontalNums = self.board.adjacent_horizontal_numbers_expanded(
                row, column)
            horizontalStr = ""

            for i in horizontalNums:  # Transform Tuple into String
                if i == None:
                    horizontalStr += "2"
                else:
                    horizontalStr += str(i)

            # Go to dictionary and get the correct play
            correctPlay = playDict[horizontalStr]

            if correctPlay != None:  # If there is a play that should be made, return it
                return correctPlay

            # Vertical
            verticalNums = self.board.adjacent_vertical_numbers_expanded(
                row, column)
            verticalStr = ""

            for i in verticalNums:  # Transform Tuple into String
                if i == None:
                    verticalStr += "2"
                else:
                    verticalStr += str(i)

            # Go to dictionary and get the correct play
            correctPlay = playDict[verticalStr]

            if correctPlay != None:  # If there is a play that should be made, return it
                return correctPlay

            return None

        # Look in all empty positions for a obligatory play to make
        for i in self.board.emptyPositions:
            obligatoryPlay = getObligatoryPlayForPosition(i[0], i[1])

            if obligatoryPlay != None:
                return finishActionLookup([self.Action(i[0], i[1], obligatoryPlay)])

        pass

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        newBoard = state.board.play(action)

        return TakuzuState(newBoard)

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        for row in range(state.board.size):
            for col in range(state.board.size):
                if state.board.get_number(row, col) == 2:
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
    print("Initial:\n", board, sep="")
