# -*- coding: utf-8 -*-
import numpy as np


class SimplexMethod():
    def __init__(self, canonical_form_problem):
        """
        Inicializa o método Simplex com um problema na forma canônica.

        Args:
        - canonical_form_problem: Objeto representando um problema na forma canônica.
        """
        self.problem = canonical_form_problem
        self.result = self.problem.x  # Resultado da solução
        self.result_cost = self.problem.get_cost()  # Custo associado à solução

    def run(self):
        """
        Executa o método Simplex para resolver o problema.

        O método continua iterando até que não haja mais custos reduzidos negativos.
        """
        self.problem.start_basis()  # Inicializa a base do problema
        basic_feasible_solution = self.problem.x  # Solução básica viável
        reduced_costs = self.problem.get_reduced_costs()  # Custos reduzidos
        has_negatives = self.has_negative_value(reduced_costs)

        while has_negatives[0]:
            negative_index_chosen = np.random.choice(has_negatives[1], 1)[0]
            j = negative_index_chosen
            self.update_basis(j)
            reduced_costs = self.problem.get_reduced_costs()
            has_negatives = self.has_negative_value(reduced_costs)

        self.print_optimal_solution()

    def update_basis(self, j):
        """
        Atualiza a base do problema com base na variável não básica escolhida.

        Args:
        - j: Índice da variável não básica a ser incluída na base.
        """
        B_inv = np.linalg.inv(self.problem.B)
        A_j = self.problem.A[:, j].reshape(self.problem.m, 1)
        u = np.dot(B_inv, A_j)

        if len(np.where(u > 0)[0]) > 0:
            self.perform_basis_change(u, j)
        else:
            self.print_suboptimal_solution()

    def perform_basis_change(self, u, j):
        """
        Executa a mudança de base com base nos vetores u e j.

        Args:
        - u: Vetor representando a coluna correspondente à variável não básica.
        - j: Índice da variável não básica.
        """
        xb = self.problem.x[self.problem.basic_index]
        theta = np.divide(xb, u, out=np.zeros_like(xb), where=u > 0)
        theta_min = np.min(theta[np.nonzero(theta)])
        theta_l_idx = np.where(theta == theta_min)[0][0]
        self.problem.changeBasis(theta_min, theta_l_idx, j, u)

    def print_optimal_solution(self):
        """ Imprime a solução ótima encontrada pelo método Simplex. """
        print('X ótimo:\n{}'.format(self.problem.x))
        print('Z ótimo: {}'.format(self.problem.get_cost()))

    def print_suboptimal_solution(self):
        """ Imprime uma mensagem indicando que a solução ótima não foi encontrada. """
        print('X ótimo:\n{}'.format(self.problem.x))
        print('Z ótimo: -inf')

    def has_negative_value(self, array):
        """
        Verifica se há valores negativos no array.

        Args:
        - array: Array a ser verificado.

        Returns:
        - Lista contendo um booleano indicando se há valores negativos e
          os índices onde esses valores negativos estão localizados.
        """
        return [len(np.where(array < 0)[0]) > 0, np.where(array < 0)[0]]
