# -*- coding: utf-8 -*-
import numpy as np


class Canonical():
    """
    Esta classe representa a modelagem de um problema de Programação Linear
    na forma canônica.
    """

    def __init__(self, c, A, b, description="Problema na forma canônica"):
        """
        Inicia um objeto da classe Canonical.

        Args:
        - c: Coeficientes da função objetivo Z.
        - A: Coeficientes das restrições do problema.
        - b: Valores do lado direito das restrições.
        - description: Descrição opcional do problema.
        """
        self.c = np.expand_dims(c, axis=0)  # Coeficientes da função objetivo Z
        self.A = A  # Coeficientes das restrições do problema
        self.b = b  # Valores do lado direito das restrições
        self.description = description  # Descrição opcional do problema
        self.m = b.shape[0]  # Número de restrições
        self.n = c.shape[0]  # Número de variáveis
        self.basic_index = np.zeros((1, self.m))  # Índices das variáveis básicas
        self.nonbasic_index = np.zeros((1, self.n - self.m))  # Índices das variáveis não básicas
        self.B = np.zeros((self.m, self.m))  # Matriz da base
        self.x = np.zeros((self.n, 1))  # Vetor solução
        self.xb = np.zeros((1, self.m))  # Valores das variáveis básicas

    def get_cost(self):
        """ Calcula o valor da função custo: (c'x). """
        return np.dot(self.c, self.x)[0]

    def start_basis(self, max_attempts=100):
        """
        Inicializa a estrutura de base do problema.

        Args:
        - max_attempts: Número máximo de tentativas para encontrar uma base não singular.
        """
        xb, cb, singular_B, attempts = self._initialize_basis(max_attempts)

        x = np.zeros((self.n, 1))
        for idx, x_i in np.ndenumerate(self.basic_index):
            x[x_i] = xb[idx[0]]

        self.x = x
        self.cb = self.c[:, self.basic_index]

    def _initialize_basis(self, max_attempts):
        """
        Tenta encontrar uma base não singular.

        Args:
        - max_attempts: Número máximo de tentativas para encontrar uma base não singular.

        Returns:
        - Tupla contendo xb, cb, singular_B, attempts.
        """
        xb = np.full((1, self.m), -1.0)
        cb = np.zeros((self.m, 1))
        singular_B = True
        attempts = 0

        while singular_B and attempts < max_attempts:
            basic_index = self._choose_basic_index()

            # Garante que basic_index está dentro dos limites
            if np.max(basic_index) < self.n:
                B = self.A[:, basic_index]

                # Verifica se B é singular
                if np.linalg.matrix_rank(B) == min(B.shape):
                    singular_B = False
                    xb = np.dot(np.linalg.inv(B), self.b)

            attempts += 1

        if attempts == max_attempts:
            raise RuntimeError("Falha ao encontrar uma base não singular dentro do número máximo de tentativas.")
            
        self.basic_index = basic_index
        self.B = B
        self.xb = xb

        return xb, cb, singular_B, attempts

    def _choose_basic_index(self):
        """ Escolhe basic_index aleatoriamente. """
        return np.random.choice(self.n, self.m, replace=False)

    def get_reduced_costs(self):
        """ Calcula os custos reduzidos para as variáveis não básicas. """
        nonbasic_index = self._compute_nonbasic_index()

        if nonbasic_index.size == 0:
            self.print_optimal_solution()
            return np.zeros((self.n, 1))

        reduced_cost = self._compute_reduced_costs(nonbasic_index)
        self.nonbasic_index = nonbasic_index
        return reduced_cost

    def _compute_nonbasic_index(self):
        """ Calcula nonbasic_index. """
        return np.setdiff1d(np.arange(self.n), self.basic_index).reshape(1, -1)

    def _compute_reduced_costs(self, nonbasic_index):
        """ Calcula os custos reduzidos para as variáveis não básicas. """
        reduced_cost = np.zeros((self.n, 1))
        for idx, nb_idx in np.ndenumerate(nonbasic_index):
            c_j = self.c[0, nb_idx]
            B_inv = np.linalg.inv(self.B)
            A_j = self.A[:, nb_idx].reshape(self.m, 1)
            reduced_cost[nb_idx] = c_j - np.dot(np.dot(self.c[0, self.basic_index], B_inv), A_j)[0]
        return reduced_cost

    def changeBasis(self, theta_min, theta_l_idx, j, u):
        """ Atualiza a base e a solução. """
        d = self._compute_direction_vector(j, u)
        y = self._compute_new_solution(theta_min, d)

        new_basis = self._update_basis(theta_l_idx, j)
        B = self.A[:, new_basis]

        xb = np.dot(np.linalg.inv(B), self.b)
        self.x = y
        self.basic_index = new_basis
        self.B = B
        self.xb = xb

    def _compute_direction_vector(self, j, u):
        """ Calcula o vetor de direção para a mudança de base. """
        d = np.zeros((self.n, 1))
        d[self.basic_index] = -1 * u
        d[j] = 1
        return d

    def _compute_new_solution(self, theta_min, d):
        """ Calcula a nova solução após a mudança de base. """
        y = self.x + theta_min * d
        return y

    def _update_basis(self, theta_l_idx, j):
        """ Atualiza a base substituindo uma variável básica. """
        new_basis = np.copy(self.basic_index)
        l = new_basis[theta_l_idx]
        new_basis[theta_l_idx] = j
        return new_basis

    def print_representation(self):
        """ Imprime no console a representação do problema. """
        pass

    def print_optimal_solution(self):
        """ Imprime a solução ótima. """
        print('X ótimo:\n{}'.format(self.x))
        print('Z ótimo: {}'.format(self.get_cost()))
