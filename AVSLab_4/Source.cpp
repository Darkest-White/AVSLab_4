#include <iostream>
#include <cmath>
#include <omp.h>
using namespace std;

double function(double x) 
{
    return log(1.1 + tan(x / 2));
}

double calculate_integral_left_rect(double a, double b, double h)
{
    double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < static_cast<int>((b - a) / h); ++i) 
    {
        double x = a + i * h;
        sum += function(x);
    }
    return h * sum;
}

double calculate_integral_right_rect(double a, double b, double h) 
{
    double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
    for (int i = 1; i <= static_cast<int>((b - a) / h); ++i) 
    {
        double x = a + i * h;
        sum += function(x);
    }
    return h * sum;
}

double calculate_integral_midpoint_rect(double a, double b, double h) 
{
    double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < static_cast<int>((b - a) / h); ++i) 
    {
        double x = a + (i + 0.5) * h;
        sum += function(x);
    }
    return h * sum;
}

double calculate_integral_trapezoidal(double a, double b, double h) {

    double sum = function(a) + function(b);
#pragma omp parallel for reduction(+:sum)
    for (int i = 1; i < static_cast<int>((b - a) / h); ++i) 
    {
        double x = a + i * h;
        sum += 2 * function(x);
    }
    return h * sum / 2;
}

double calculate_integral_simpson(double a, double b, double h) 
{
    double sum = function(a) + function(b);
#pragma omp parallel for reduction(+:sum)
    for (int i = 1; i < static_cast<int>((b - a) / (2 * h)); ++i) 
    {
        double x = a + 2 * i * h;
        sum += 4 * function(x);
    }
#pragma omp parallel for reduction(+:sum)
    for (int i = 1; i < static_cast<int>((b - a) / (2 * h) + 1); ++i) {
        double x = a + (2 * i - 1) * h;
        sum += 2 * function(x);
    }
    return h * sum / 3;
}

void calculate_integral_parallel(double a, double b, double precision) 
{
    double h = 0.2;

    double integral_left_rect = calculate_integral_left_rect(a, b, h);
    double integral_right_rect = calculate_integral_right_rect(a, b, h);
    double integral_midpoint_rect = calculate_integral_midpoint_rect(a, b, h);
    double integral_trapezoidal = calculate_integral_trapezoidal(a, b, h);
    double integral_simpson = calculate_integral_simpson(a, b, h);

    cout << "Integral (Left Rectangular): " << integral_left_rect << endl;
    cout << "Integral (Right Rectangular): " << integral_right_rect << endl;
    cout << "Integral (Midpoint Rectangular): " << integral_midpoint_rect << endl;
    cout << "Integral (Trapezoidal): " << integral_trapezoidal << endl;
    cout << "Integral (Simpson): " << integral_simpson << endl;
}

int main() 
{
    double a = 0.0;
    double b = 2.0;
    double precision = 1e-7;

    double start_time = omp_get_wtime();

    calculate_integral_parallel(a, b, precision);

    double end_time = omp_get_wtime();

    cout << "Execution time: " << end_time - start_time << " seconds" << endl;

    return 0;
}