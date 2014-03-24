// gmdh.cpp : Defines the entry point for the console application.
//

// #define debug_print   // отрадочная печать
// #define using_alglib  // для правильного решения вырожденных СЛУ

#include "stdafx.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

#ifdef using_alglib
#include "ap.h"
#include "solvers.h"
using namespace alglib;
#endif

#define WEIGHTS_CONST 4
#define MAX_MODELS_NUM 10
#define MAX_ROW_NUM 50
#define PERCENT 0.4

class Model {
public:
  int id;
  int input1;
  int input2;
  double *weights;
  double CR;

public:
  Model(){}

  Model(int id, int input1, int input2){
    this->id = id;
    this->input1 = input1;
    this->input2 = input2;
    weights = new double[WEIGHTS_CONST];
    this->CR = 0.0;
  }

  void setWeights(double *w){
    for(int i = 0; i < WEIGHTS_CONST; i++)
      weights[i] = w[i];
  }

  void printWeights(){
    std::cout<<"weights = "<<weights[0]<<"; "<<weights[1]<<"; "<<weights[2]<<"; "<<weights[3]<<std::endl;
  }

  double f(double xi, double xj){
    return weights[0] + weights[1] * xi + weights[2] * xj + weights[3] * xi * xj;
  }

  ~Model()
  {
    delete []weights;
  }
};

class Row {
public:
  int L;           // число моделей
  Model **models;  // массив моделей
  double CR;       // ошибка ряда

public:
  Row(){}

  Row(int num_of_models){
    L = num_of_models;
    models = new Model * [L];
    CR = 0.0;
  }

  void countCR(){
    // ошибка ряда = min_error
    CR = models[0]->CR;
    for(int i = 0; i < L; i++){
      if(CR > models[i]->CR) CR = models[i]->CR;
    }
  }

  ~Row()
  {
    for (int i = 0; i < L; i++)
      delete models[i];
    delete []models;
  }
};

class GMDH {
public:
  int m;                 // число признаков
  int N;                 // число образцов выборки (N = NA + NB)
  int NA;
  int NB;
  double percentage;     // процент отбираемых "лучших" моделей
  std::vector<Row *> rows;

public:
  GMDH(int m, int N, int NA){
    this->m = m;
    this->N = N;
    this->NA = NA;
    this->NB = N - NA;
    percentage = PERCENT;
  }

  GMDH(char *filename){
    std::ifstream fin(filename);
    fin>>m>>N>>NA;
    NB = N - NA;
    percentage = PERCENT;
    int sz;
    fin>>sz;

    int L; // число моделей текущего ряда
    double CR; // ошибка ряда

    int m_id, m_input1, m_input2;
    double mCR;
    for(int it = 0; it < sz; it++){
      fin>>L>>CR;
      Row * row = new Row(L);    // создаем новый ряд
      row->CR = CR;
      for(int l = 0; l < L; l++){
        // добавляем модель в ряд
        fin>>m_id>>m_input1>>m_input2>>mCR;
        row->models[l] = new Model(m_id, m_input1, m_input2);

        double *weights = new double[WEIGHTS_CONST];
        for(int i = 0; i < WEIGHTS_CONST; i++){
          fin>>weights[i];
        }
        row->models[l]->setWeights(weights);
        delete []weights;
        row->models[l]->CR = mCR;
      }
      rows.insert(rows.end(), row);
    }
    fin.close();
  }

  void saveGMDH(char *filename){
    std::ofstream fout(filename);
    fout<<m<<" "<<N<<" "<<NA<<std::endl;
    int L;  // число моделей текущего ряда

    // формирование рядов
    std::vector<Row>::iterator row_it;
    std::vector<Row>::size_type sz = rows.size();

    // print size of rows
    fout<<sz<<std::endl;
    for(int it = 0; it < sz; it++){
      L = rows[it]->L;
      fout<<L<<" "<<rows[it]->CR<<std::endl;
      for(int l = 0; l < L; l++){
        fout<<rows[it]->models[l]->id;
        fout<<" "<<rows[it]->models[l]->input1;
        fout<<" "<<rows[it]->models[l]->input2;
        fout<<" "<<rows[it]->models[l]->CR;
        for(int i = 0; i < WEIGHTS_CONST; i++){
          fout << " " << rows[it]->models[l]->weights[i];
        }
        fout<<std::endl;
      }
    }
    fout.close();
  }

  // число сочетаний из n по 2 = C(n, 2)
  int comb2(int n){
    if(n == 0 || n == 1) return 0;
    int comb = 0;
    for(int i = 0; i < n; i++){
      for(int j = i+1; j < n; j++){
        comb += 1;
      }
    }
    return comb;
  }

  /* Решение СЛУ методом Гаусса (Жордана) для матрицы [n][n+1]
  где, посл столбец - столбец своб членов */
  double* solveLinearEquasion(double **A_matr, double *b_matr, int n){
    int i, j, k;
    double u;
    double *x = new double[n];

    // A_matr[n][n]
    // B_matr[n]
    // z[n][n+1] = A_matr B_matr

#ifdef using_alglib
    alglib::real_2d_array a;
    a.setlength(n, n);

    for(i = 0; i < n; i++){
      for(j = 0; j < n; j++){
        a(i,j) = A_matr[i][j];
      }
    }

    alglib::real_1d_array b;
    b.setlength(n);
    for(i = 0; i < n; i++)
      b(i) = b_matr[i];

    alglib::ae_int_t info;
    alglib::densesolverreport rep;
    
    alglib::real_1d_array dlib_x;
    rmatrixsolve(a, n, b, info, rep, dlib_x);

    if (info == -3)
	{
		alglib::densesolverlsreport reps;

		rmatrixsolvels(a, n, n, b, 0.0, info, reps, dlib_x);
	}

    for(i = 0; i < n; i++)
      x[i] = dlib_x(i);
#else
    double **z = new double*[n];
    for(i = 0; i < n; i++){
      z[i] = new double[n+1];
    }
    for(i = 0; i < n; i++){
      for(j = 0; j < n; j++){
        z[i][j] = A_matr[i][j];
      }
    }
    for(i = 0; i < n; i++){
      z[i][n] = b_matr[i];
    }

    // приведение матрицы к н. треуг. виду (по Гауссу)
    for(j = 0; j < n+1; j++){
      for(i = j+1; i < n; i++){
        u = -z[i][j]/z[j][j];
        for(k = j; k < n+1; k++)
          z[i][k] = z[i][k] + u * z[j][k];
      }
    }

    // приведение правой матрицы к в. треуг. виду (по Жордану)
    for(j = n-1; j > 0; j--)
      for(i = j-1; i >= 0; i--){
        u = - z[i][j]/z[j][j];
        for(k = j; k < n+1; k++)
          z[i][k] = z[i][k] + u * z[j][k];
      }

      for(i = 0; i < n; i++){
        x[i] = z[i][n]/z[i][i];
      }
#endif

      return x; // вектор решений СЛУ
  }

  void training(double **input_data, double *y0){
    // формирование рядов
    int F = m;
    double prev_error = DBL_MAX;

    double **inputs = new double*[N];
    for(int i = 0; i < N; i++)
      inputs[i] = new double[m];

    for(int i = 0; i < N; i++){
      for(int j = 0; j < m; j++){
        inputs[i][j] = input_data[i][j];
      }
    }
    double **outputs;
    int row_number = 0;

    while(row_number < MAX_ROW_NUM){
      int l = 0;
      row_number += 1;
      std::cout << "### Row number " << row_number << std::endl;

      int L = comb2(F);    // число моделей текущего ряда = C(F, 2)
      Row * row = new Row(L);    // создаем новый ряд

      int **combinations = new int*[L]; // (model_num, i, j)
      for(int i = 0; i < L; i++)
        combinations[i] = new int[3];

      l = 0;
      for(int i = 0; i < F; i++){
        for(int j = i+1; j < F; j++){
          combinations[l][0] = l; // model_num
          combinations[l][1] = i;
          combinations[l][2] = j;
          l++;
        }
      }

      // смотрим на обучающую выборку и вычисляем весовые коэффициенты
      for(l = 0; l < L; l++){
        int model_num = combinations[l][0];
        int i = combinations[l][1];
        int j = combinations[l][2];

        // по всем сочетаниям i и j
        double **A_matr = new double*[WEIGHTS_CONST];
        for(int it = 0; it < WEIGHTS_CONST; it++)
          A_matr[it] = new double[WEIGHTS_CONST];
        for(int it1 = 0; it1 < WEIGHTS_CONST; it1++)
          for(int it2 = 0; it2 < WEIGHTS_CONST; it2++)
            A_matr[it1][it2] = 0.0;

        double *b_matr = new double[WEIGHTS_CONST];
        for(int it = 0; it < WEIGHTS_CONST; it++)
          b_matr[it] = 0.0;

        for(int k = 0; k < NA; k++){
          double xi = inputs[k][i];
          double xj = inputs[k][j];

          A_matr[0][0] += 1.0;
          A_matr[1][0] += xi;
          A_matr[2][0] += xj;
          A_matr[3][0] += xi * xj;

          A_matr[0][1] += xi;
          A_matr[1][1] += xi*xi;
          A_matr[2][1] += xi * xj;
          A_matr[3][1] += xi*xi * xj;

          A_matr[0][2] += xj;
          A_matr[1][2] += xi * xj;
          A_matr[2][2] += xj*xj;
          A_matr[3][2] += xi * xj*xj;

          A_matr[0][3] += xi * xj;
          A_matr[1][3] += xi*xi * xj;
          A_matr[2][3] += xi * xj*xj;
          A_matr[3][3] += xi*xi * xj*xj;

          b_matr[0] += y0[k];
          b_matr[1] += y0[k] * xi;
          b_matr[2] += y0[k] * xj;
          b_matr[3] += y0[k] * xi * xj;
        }

        double *weights = solveLinearEquasion(A_matr, b_matr, WEIGHTS_CONST);

        // добавляем модель в ряд
        row->models[l] = new Model(model_num, i, j);
        row->models[l]->setWeights(weights);
#ifdef debug_print
        row->models[l]->printWeights();
#endif
        delete []weights;
      }

      // вычисление ошибки для каждой модели
      for(l = 0; l < L; l++){
        int model_num = combinations[l][0];
        int i = combinations[l][1];
        int j = combinations[l][2];

        row->models[model_num]->CR = 0.0;
        // по всем образцам из проверочной выборки
        for(int k = NA; k < N; k++){
          double xi = inputs[k][i];
          double xj = inputs[k][j];
          row->models[model_num]->CR += (row->models[model_num]->f(xi, xj) - y0[k])*(row->models[model_num]->f(xi, xj) - y0[k]);
          //std::cout<<"f(xi, xj) = "<<row->models[model_num]->f(xi, xj)<<std::endl;
        }
        row->models[model_num]->CR /= NB;
        //std::cout<<"row->models["<<model_num<<"].CR "<<row->models[model_num]->CR<<std::endl;
      }

#ifdef debug_print
      std::cout<<"models before sorting:"<<std::endl;
      for(l = 0; l < L; l++)
        std::cout<<row->models[l]->id<<" "<<row->models[l]->input1<<" "<<row->models[l]->input2<<" "<<row->models[l]->CR<<std::endl;
#endif

      row->countCR(); // вычислить ошибку ряда

      // сортируем модели по возрастанию ошибки
      bool changes = true;
      while(changes){
        changes = false;
        for(l = 0; l < L-1; l++){
          if(row->models[l]->CR > row->models[l+1]->CR){
            std::swap(row->models[l]->CR, row->models[l+1]->CR);
            std::swap(row->models[l]->id, row->models[l+1]->id);
            std::swap(row->models[l]->input1, row->models[l+1]->input1);
            std::swap(row->models[l]->input2, row->models[l+1]->input2);

            for(int i = 0; i < WEIGHTS_CONST; i++)
              std::swap(row->models[l]->weights[i], row->models[l+1]->weights[i]);

            changes = true;
          }
        }
      }

#ifdef debug_print
      std::cout<<"models after sorting:"<<std::endl;
      for(l = 0; l < L; l++)
        std::cout<<row->models[l]->id<<" "<<row->models[l]->input1<<" "<<row->models[l]->input2<<" "<<row->models[l]->CR<<std::endl;
#endif

      F = int(percentage * L); // число отбираемых лучших моделей
      if(F > MAX_MODELS_NUM) F = MAX_MODELS_NUM;
      std::cout<<"L = "<<L<<std::endl;
      std::cout<<"F = "<<F<<std::endl;

      // формируем вектор выходов
      outputs = new double*[N];
      for(int k = 0; k < N; k++){
        outputs[k] = new double[F];
        for(l = 0; l < F; l++){
          double xi = inputs[k][row->models[l]->input1];
          double xj = inputs[k][row->models[l]->input2];
          outputs[k][l] = row->models[l]->f(xi, xj);
        }
      }

      for(int k = 0; k < N; k++){
        delete []inputs[k];
      }
      delete []inputs;

      for(int i = 0; i < L; i++)
        delete []combinations[i];
      delete []combinations;

      if(prev_error < row->CR){
        std::cout<<"overfitting"<<std::endl;
        std::cout<<"row_number = "<<row_number<<"; row->CR = "<<row->CR<<std::endl;
        for(int i = 0; i < N; i++){
          delete []outputs[i];
        }
        delete []outputs;
        return;
      }

      prev_error = row->CR;

      //rows.insert(rows.end(), row);
      rows.push_back(row);

      if(F <= 1){
        std::cout<<"no more models"<<std::endl;
        std::cout<<"row_number = "<<row_number<<"; row->CR = "<<row->CR<<std::endl;
        for(int i = 0; i < N; i++){
          delete []outputs[i];
        }
        delete []outputs;
        return;
      }

      inputs = outputs;
    }
    for(int i = 0; i < N; i++){
      delete []outputs[i];
    }
    delete []outputs;
  }

  double testGMDH(double *x){
    double *inputs = new double[m];
    for(int i = 0; i < m; i++)
      inputs[i] = x[i];
    int L;
    double *outputs;

    // формирование рядов
    std::vector<Row>::iterator row_it;
    std::vector<Row>::size_type sz = rows.size();
    for(int it = 0; it < sz; it++){
      // формируем вектор выходов
      L = rows[it]->L;
      outputs = new double[L];
      for(int l = 0; l < L; l++){
        double xi = inputs[rows[it]->models[l]->input1];
        double xj = inputs[rows[it]->models[l]->input2];
        outputs[l] = rows[it]->models[l]->f(xi, xj);
      }
      delete []inputs;
      inputs = outputs;
    }

    double t_val = inputs[0];
    delete []inputs;
    return t_val;
  }

  ~GMDH()
  {
    for (size_t i = 0; i < rows.size(); i++)
      delete this->rows[i]; 
  }
};


