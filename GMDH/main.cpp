#include "gmdh.cpp"

int main (void) {
  int N = 19;
  int NA = 10;
  int m = 11;
  //double y0[16] = {1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0};

//std::ifstream fin("datafile.txt");
std::ifstream fin("samples.txt");

  double **input_data = new double*[N];
  for(int i = 0; i < N; i++)
    input_data[i] = new double[m];

  for(int i = 0; i < N; i++){
    for(int j = 0; j < m; j++){
      fin>>input_data[i][j];
    }
  }

  fin.close();

std::ifstream fin2("classes.txt");

  double *y0 = new double[m];

  for(int i = 0; i < m; i++){
      fin2>>y0[i];
  }

  fin2.close();

  GMDH gmdh1 = GMDH(m, N, NA);
  // настройка весовых коэффициентов
  gmdh1.training(input_data, y0);
  // сохранить GMDH в файл
  gmdh1.saveGMDH("gmdh.dat");

  /*
  структура файла "gmdh.dat" :
  | m N NA
  | rows_size
  {rows_size} X |                   | num_of_models CR
  | {num_of_models} X | id input1 input2 CR weithts[0] weithts[1] weithts[2] weithts[3]
  */

  // загрузить GMDH из файла
  GMDH gmdh2 = GMDH("gmdh.dat");
  gmdh2.saveGMDH("gmdh2.dat"); // проверочка

  double test1[11] = {0.76500000000000001, 0.52000000000000002, 0.17000000000000001, -0.16500000000000001, -0.36499999999999999, -0.435, -0.42499999999999999, -0.37, -0.33000000000000002, -0.32500000000000001, -0.33500000000000002};

  std::cout<<"Normal: "<<gmdh2.testGMDH(test1)<<std::endl;
  //std::cout<<"High: "<<gmdh2.testGMDH(test2)<<std::endl;
  //std::cout<<"High: "<<gmdh2.testGMDH(test3)<<std::endl;


  system ("pause"); /* execute M$-DOS' pause command */
}
