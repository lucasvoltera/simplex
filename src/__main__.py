# -*- coding: utf-8 -*-
import numpy as np
from utils.canonical import Canonical
from utils.simplex_method import SimplexMethod

if __name__ == '__main__':
    
    custos = np.array([20, 10, 25, 8, 10, 7, 12, 15, 5])
    proteinas = np.array([18, 6, 22, 7, 2, 8, 2, 4, 0])
    carboidratos = np.array([2, 1, 5, 73, 20, 50, 8, 4, 0])
    gorduras = np.array([3, 5, 10, 0.6, 0.5, 1.5, 15, 18, 14])
    restricoes = np.array([150, 300, 50])

    # Coeficientes da função objetivo
    c = np.transpose(np.array(custos))

    # Coeficientes das restrições (proteínas, carboidratos, gorduras)
    # Proteinas: Peito de Frango, Ovos, Carne Vermelha, 
    # Carboidrato: Arroz, Batata, Pão integral, 
    # Gordura: Abacate, Castanha-do-pará, Azeite
    A = np.array([proteinas, carboidratos, gorduras])

    # Termos independentes das restrições
    b = np.transpose(np.array(restricoes))

    print(f'Funcao objetivo: Min({c})')
    print(f'Quantidade de macronutrientes de cada alimento:\nProteina:{A[0]} >= {b[0]}\nCarboidrato:{A[1]} => {b[1]}\nGordura:{A[2]} => {b[2]}')
    # print(f'Restrições de macronutrientes: {b}')

    print("\nResolvendo o problema com o Simplex: ")
    # Definir instância do Simplex
    np.random.seed(42)
    problem = Canonical(c, A, b, "Min 20x1 + 10x2 + 25x3 + 8x4 + 10x5 + 7x6 + 12x7 + 15x8 + 5x9")
    simplex = SimplexMethod(problem)
    simplex.run()

