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

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de Takuzu."""
    
    def __init__(self, size, board):
        self.size = size
        self.board = board

    def __str__(self):
        output = ""
        #for line in self.board:
        #    output += ("\t".join(line)+"\n")
        for row in range(self.size):
            for col in range(self.size-1):
                output += str(Board.get_number(self, row, col))+"\t"
            output += str(Board.get_number(self, row, self.size))+"\n"
        return output

    def get_number(self, row: int, col: int) -> int:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[row-1][col-1]

    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        below, above = -1, -1
        if row < self.size:
            below = Board.get_number(self, row+1, col)
        if row > 0:
            above = Board.get_number(self, row-1, col)
        return (below, above)

    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        left, right = -1, -1
        if col < self.size:
            right = Board.get_number(self, row, col+1)
        if col > 0:
            left = Board.get_number(self, row, col-1)
        return (left, right)

    @staticmethod
    def parse_instance_from_stdin():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 takuzu.py < input_T01

            > from sys import stdin
            > stdin.readline()
        """
        # This can probably be done with just numpy arrays but whatever
        size = int(sys.stdin.readline().split('\n')[0])
        board = []
        for _ in range(size):
            linha = sys.stdin.readline().replace('\n', '\t').split('\t')[:-1]
            linha = [int(value) for value in linha]
            board += [np.array(linha)]
        board = np.array(board)
        return Board(size, board)

    # TODO: outros metodos da classe


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        # TODO
        pass

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO
        pass

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO
        pass

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        # TODO
        pass

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
