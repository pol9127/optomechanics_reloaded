/* Script used for the nonlinear calibration of particle position.
Takes as input the name of the compiled executable, the input file (where the distorted vector is saved),
the output file (where to save the corrected vector), number of elements to process and list of nonlinear coefficients.
@ Author: Andrei Militaru
@ date: 11th of June 2019 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <string>

using std::cout;
using std::endl;
using std::cin;
using std::pow;
using std::abs;

int main (int argc, char* argv[]) {
    
    assert(argc < 20);
    assert(argc > 5);
    
    std::ifstream read_file;
    read_file.open(argv[1]);
    assert(read_file.is_open());
    
    std::ofstream write_file;
    write_file.open(argv[2]);
    assert(write_file.is_open());
    write_file.precision(12);
    
    double c[20];
    int len = atoi(argv[3]);
    double tolerance = atof(argv[4]);
    int max_iter = atoi(argv[5]);
    
    for (int i = 6; i < argc; i++) {
        c[i-6] = atof(argv[i]);
    }
    
    int iterations = 0;
    double input;
    double current_value;
    double err;
    double f, f_der;
    double z0;
    
    for (int j = 0; j < len; j++) {
        
        z0 = 0.0;
        err = 1.0;
        iterations = 0;
        read_file >> input;
        do {
            current_value = z0;
            f = input - z0;
            f_der = -1;
            for (int i = 0; i < argc-6; i++) {
                f -= c[i]*pow(z0,i+2);
                f_der -= (i+2)*c[i]*pow(z0,i+1);
            }
            assert(f_der != 0.0);
            z0 -= f/f_der;
            err = abs(z0 - current_value);
        } 
        while ((err > tolerance) && (++iterations < max_iter));
        write_file << z0 << "\n";
    }
    
    read_file.close();
    write_file.close();
    return 0;
}