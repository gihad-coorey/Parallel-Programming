#include <getopt.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Default values if not specified in command line
#define NUM_THREADS_DEFAULT 4
#define DENSITY_DEFAULT 0.02
#define SIZE_DEFAULT 10

/**
 * Matrix structure.
 * Example 2x3 matrix:
 * 1 2 3
 * 4 5 6
 */
typedef struct {
  char *name;
  int rows;
  int cols;
  int density;
  int **data;
} Matrix;

/**
 * Row-compressed Matrix structure
 * Given original Matrix:
 * 1  0  2
 * 0  0  0
 * 0  4  0
 *
 * CompressedMatrix `values`:
 * 1  2
 * 0  0
 * 4
 * CompressedMatrix `col_indices`:
 * 0  2
 * 0  0
 * 1
 */
typedef struct {
  char *name;
  int rows;
  int max_cols;
  int *row_sizes; // Stores the number of non-zero values in each row
  int **values;   // B matrix: Stores the non-zero values of the matrix
  int **
      col_indices; // C matrix: Stores the column indices of the non-zero values
} CompressedMatrix;

void parse_options(int argc, char *argv[], int *num_threads, double *density,
                   int *size, char **output_file, int *compare_flag, int *help);
void print_usage_help();
void check_allocation(void *ptr, char *msg);

Matrix *generate_sparse_matrix(int rows, int cols, double density, char *name,
                               int seed);
Matrix *initialise_matrix(int rows, int cols, char *name);
CompressedMatrix *initialise_compressed_matrix(int rows, int cols);
CompressedMatrix *compress_matrix(Matrix *matrix);
Matrix *uncompress_matrix(CompressedMatrix *compressed_matrix);
CompressedMatrix *
gather_partial_compressed_matrices(CompressedMatrix *partial_compressed_Y,
                                   int size, int num_processes, int rank);
Matrix *
gather_partial_uncompressed_matrices(Matrix *partial_uncompressed_result,
                                     int size, int num_processes, int rank);

int *serialise_compressed_matrix(CompressedMatrix *matrix);
CompressedMatrix *deserialise_compressed_matrix(int *serialised_matrix,
                                                int rows, int cols);
int *serialise_uncompressed_matrix(Matrix *matrix);
Matrix *deserialise_uncompressed_matrix(int *serialised_matrix, int rows,
                                        int cols);
int get_serialised_size(CompressedMatrix *matrix);

Matrix *multiply_compressed_matrices(CompressedMatrix *matrix_X,
                                     CompressedMatrix *matrix_Y);
Matrix *multiply_uncompressed_matrices(Matrix *matrix_X, Matrix *matrix_Y);

int compare_uncompressed_matrices(Matrix *matrix1, Matrix *matrix2);
void write_compressed_matrix_to_file(FILE *fileB, FILE *fileC,
                                     CompressedMatrix *matrix);
void write_uncompressed_matrix_to_file(FILE *file, Matrix *matrix);
void free_matrix(Matrix *matrix);
void free_compressed_matrix(CompressedMatrix *matrix);

int main(int argc, char *argv[]) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int rank, num_processes;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  if(provided != MPI_THREAD_FUNNELED) {
    fprintf(stderr, "MPI_Init_thread did not provide the requested threading level.\n");
    MPI_Finalize();
    return EXIT_FAILURE;
  } else {
    printf("MPI_Init_thread provided the requested threading level in process %d.\n", rank);
  }

  int num_threads, size, compare_flag, help;
  double density;
  char *output_filename;

  parse_options(argc, argv, &num_threads, &density, &size, &output_filename,
                &compare_flag, &help);

  // Matrix size must be an integer multiple of `num_processes`
  if (size % num_processes != 0) {
    fprintf(stderr,
            "Requested matrix size must be divisible by the number of "
            "processes %i.\n",
            num_processes);
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  // Print help message and exit
  if (help) {
    print_usage_help();
    MPI_Finalize();
    return EXIT_SUCCESS;
  }

  // Set number of threads for OpenMP
  if (num_threads > omp_get_max_threads()) {
    fprintf(stderr,
            "\033[1;33mWarning: Requested number of threads is greater than "
            "the maximum "
            "number of cores available (%i) to processes %i. Performance "
            "effects are unknown.\033[0m\n",
            omp_get_max_threads(), rank);
  }
  omp_set_num_threads(num_threads);

  // Print configuration
  printf("MPI Matrix Multiplication running with %d process(es) and %d "
         "thread(s) each\n",
         num_processes, num_threads);
  printf("Hello from process %d.\n", rank);
  printf("Matrix size (square): %d, density: %f\n", size, density);
  printf("Output file: %s\n", output_filename ? output_filename : "None");
  printf(
      "Compare compressed result to standard algorithm for correctness?: %s\n",
      compare_flag ? "Yes" : "No");

  // Generate matrices
  // Because both matrices are seeded by the same rank, they will be the same,
  // but we will pretend they are different to avoid oversimplifying the
  // optimisation
  int rows_per_proc = size / num_processes;

  Matrix *partial_X =
      generate_sparse_matrix(rows_per_proc, size, density, "X", rank);
  Matrix *partial_Y =
      generate_sparse_matrix(rows_per_proc, size, density, "Y", rank);

  // Compress matrices
  CompressedMatrix *partial_compressed_X = compress_matrix(partial_X);
  CompressedMatrix *partial_compressed_Y = compress_matrix(partial_Y);

  // Gather compressed partial_Y from all processes into complete_compressed_Y
  CompressedMatrix *complete_compressed_Y = gather_partial_compressed_matrices(
      partial_compressed_Y, size, num_processes, rank);

  // Multiply partial_compressed_X and complete_compressed_matrix_Y to get
  // partial_uncompressed_result
  Matrix *partial_uncompressed_result =
      multiply_compressed_matrices(partial_compressed_X, complete_compressed_Y);

  // Allgather partial_uncompressed_result into complete_uncompressed_result
  Matrix *complete_uncompressed_result = gather_partial_uncompressed_matrices(
      partial_uncompressed_result, size, num_processes, rank);

  if (compare_flag) {

    Matrix *complete_X, *complete_Y;
// Allgather partial_X and partial_Y into complete_X and complete_Y
#pragma omp parallel sections default(shared)
    {
#pragma omp section
      {
        complete_X = gather_partial_uncompressed_matrices(partial_X, size,
                                                          num_processes, rank);
      }
#pragma omp section
      {
        complete_Y = gather_partial_uncompressed_matrices(partial_Y, size,
                                                          num_processes, rank);
      }
    }

    Matrix *standard_algorithm_result =
        multiply_uncompressed_matrices(complete_X, complete_Y);

    if (rank == 0) {
#pragma omp parallel sections default(shared)
      {
#pragma omp section
        {
          printf("Comparing standard algorithm result to compressed "
                 "multiplication result...\n");
          if (compare_uncompressed_matrices(complete_uncompressed_result,
                                            standard_algorithm_result)) {
            printf("Results are equal!\n");
          } else {
            printf("Results are different!\n");
          };
        }
#pragma omp section
        {
          FILE *file = fopen("standard_algorithm_result.txt", "w");
          standard_algorithm_result->name = "XY-B/XY-C complete";
          write_uncompressed_matrix_to_file(file, standard_algorithm_result);
          fclose(file);
          free_matrix(standard_algorithm_result);
        }
#pragma omp section
        {
          complete_uncompressed_result->name = "XY-B/XY-C complete";
          FILE *file2 = fopen("compressed_algorithm_result.txt", "w");
          write_uncompressed_matrix_to_file(file2,
                                            complete_uncompressed_result);
          fclose(file2);
        }
      }
    }
  }

  if (output_filename) {
    // Remove trailing newline character from output_filename
    size_t len = strlen(output_filename);
    if (output_filename[len - 1] == '\n') {
      output_filename[len - 1] = '\0';
      printf("Output filename: %s\n", output_filename);
    }
    FILE *output_file = strcmp(output_filename, "stdout")
                            ? fopen(output_filename, "w")
                            : stdout;

    // Allgather partial_compressed_X from all processes
    // Could use allgather here instead of allgatherv, but we'd have to write a
    // new but very similar helper function to
    // `gather_partial_compressed_matrices`, so we'll just use allgatherv for
    // now
    CompressedMatrix *complete_compressed_X =
        gather_partial_compressed_matrices(partial_compressed_X, size,
                                           num_processes, rank);

    if (rank == 0) {
      // Compress result matrix to write to file
      CompressedMatrix *result_compressed_matrix =
          compress_matrix(complete_uncompressed_result);

      complete_compressed_X->name = "X-B/X-C complete";
      complete_compressed_Y->name = "Y-B/Y-C complete";
      result_compressed_matrix->name = "XY-B/XY-C complete";

      write_compressed_matrix_to_file(output_file, output_file,
                                      complete_compressed_X);
      write_compressed_matrix_to_file(output_file, output_file,
                                      complete_compressed_Y);

      write_compressed_matrix_to_file(output_file, output_file,
                                      result_compressed_matrix);
    }
  }

  MPI_Finalize();
  printf("Finalised MPI.\n");
  return EXIT_SUCCESS;
}

void parse_options(int argc, char *argv[], int *num_threads, double *density,
                   int *size, char **output_filename, int *compare, int *help) {
  int c;
  *num_threads = NUM_THREADS_DEFAULT;
  *density = DENSITY_DEFAULT;
  *size = SIZE_DEFAULT;
  *output_filename = NULL;
  *compare = 0;
  *help = 0;

  while ((c = getopt(argc, argv, "t:d:s:o:ch")) != -1) {
    switch (c) {
    case 't':
      *num_threads = atoi(optarg);
      if (*num_threads < 1) {
        fprintf(stderr, "Number of threads must be greater than 0.\n");
        exit(EXIT_FAILURE);
      }
      break;
    case 'd':
      *density = atof(optarg);
      if (*density <= 0 || *density > 1) {
        fprintf(stderr, "Density must be between 0 and 1.\n");
        exit(EXIT_FAILURE);
      }
      break;
    case 's':
      *size = atoi(optarg);
      if (*size < 1) {
        fprintf(stderr, "Matrix size must be greater than 0.\n");
        exit(EXIT_FAILURE);
      }
      break;
    case 'o':
      *output_filename = optarg;
      break;
    case 'c':
      *compare = 1;
      break;
    case 'h':
      *help = 1;
      break;
    }
  }
}

void print_usage_help() {
  printf("Usage: [-t num_threads] [-d density] [-s size] [-o "
         "output_file] [-c] [-h]\n");
  printf("Options:\n");
  printf("  -t <num_threads>    Number of threads to use for OpenMP\n");
  printf("  -d <density>        Density of the generated matrices\n");
  printf("  -s <size>           Size of the square matrices\n");
  printf("  -o [<output_file>]  Output file to write compressed matrices. "
         "Pass `stdout` to print to stdout.\n");
  printf("  -c                  Compare compressed result to standard "
         "algorithm\n");
  printf("  -h                  Print this help message\n");
}

void check_allocation(void *ptr, char *msg) {
  if (ptr == NULL) {
    fprintf(stderr, "Failed to allocate memory: %s\n", msg);
    exit(EXIT_FAILURE);
  }
}

Matrix *initialise_matrix(int rows, int cols, char *name) {
  Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
  check_allocation(matrix, "initialise_matrix: matrix");
  matrix->rows = rows;
  matrix->cols = cols;
  matrix->data = (int **)calloc(rows, sizeof(int *));
  check_allocation(matrix->data, "initialise_matrix: matrix->data");
  for (int i = 0; i < rows; i++) {
    matrix->data[i] = (int *)calloc(cols, sizeof(int));
    check_allocation(matrix->data[i], "initialise_matrix: matrix->data[i]");
  }
  matrix->name = name;
  return matrix;
}

Matrix *generate_sparse_matrix(int rows, int cols, double density, char *name,
                               int seed) {
  Matrix *matrix = initialise_matrix(rows, cols, name);

  srand(seed);

#pragma omp parallel for default(shared) schedule(runtime)
  for (int i = 0; i < matrix->rows; i++) {
    for (int j = 0; j < matrix->cols; j++) {
      double prob = (double)rand() / RAND_MAX;
      if (prob <= density) {
        matrix->data[i][j] = rand() % 10 + 1;
      } else {
        matrix->data[i][j] = 0;
      }
    }
  }

  return matrix;
}

CompressedMatrix *initialise_compressed_matrix(int rows, int cols) {
  CompressedMatrix *compressed_matrix =
      (CompressedMatrix *)malloc(sizeof(CompressedMatrix));
  check_allocation(compressed_matrix, "initialise_compressed_matrix");
  compressed_matrix->rows = rows;
  compressed_matrix->max_cols = cols;
  compressed_matrix->row_sizes = (int *)calloc(rows, sizeof(int));
  check_allocation(
      compressed_matrix->row_sizes,
      "initialise_compressed_matrix: compressed_matrix->row_sizes");
  compressed_matrix->values = (int **)calloc(rows, sizeof(int *));
  check_allocation(compressed_matrix->values,
                   "initialise_compressed_matrix: compressed_matrix->values");
  compressed_matrix->col_indices = (int **)calloc(rows, sizeof(int *));
  check_allocation(
      compressed_matrix->col_indices,
      "initialise_compressed_matrix: compressed_matrix->col_indices");
  return compressed_matrix;
}

CompressedMatrix *compress_matrix(Matrix *matrix) {
  CompressedMatrix *compressed_matrix =
      initialise_compressed_matrix(matrix->rows, matrix->cols);

#pragma omp parallel for default(shared) schedule(runtime)
  for (int i = 0; i < matrix->rows; i++) {
    int non_zeroes_in_row = 0;

    // Initially allocate space for the maximum possible elements in the row
    // (i.e., matrix->cols)
    compressed_matrix->values[i] = (int *)calloc(matrix->cols, sizeof(int));
    check_allocation(compressed_matrix->values[i],
                     "compress_matrix: compressed_matrix->values[i]");
    compressed_matrix->col_indices[i] =
        (int *)calloc(matrix->cols, sizeof(int));
    check_allocation(compressed_matrix->col_indices[i],
                     "compress_matrix: compressed_matrix->col_indices[i]");

    for (int j = 0; j < matrix->cols; j++) {
      if (matrix->data[i][j] != 0) {
        // Store the non-zero value and its column index
        compressed_matrix->values[i][non_zeroes_in_row] = matrix->data[i][j];
        compressed_matrix->col_indices[i][non_zeroes_in_row] = j;
        non_zeroes_in_row++;
      }
    }

    // If there are no non-zero elements, initialize the first two elements to 0
    if (non_zeroes_in_row == 0) {
      compressed_matrix->values[i][0] = 0;
      compressed_matrix->col_indices[i][0] = 0;
      compressed_matrix->values[i][1] = 0;
      compressed_matrix->col_indices[i][1] = 0;
      non_zeroes_in_row = 2; // Set non_zeroes_in_row to 2 to avoid reallocation
    } else if (non_zeroes_in_row < matrix->cols) {
      // Reallocate to fit exactly the number of non-zero elements
      compressed_matrix->values[i] = (int *)realloc(
          compressed_matrix->values[i], non_zeroes_in_row * sizeof(int));
      check_allocation(compressed_matrix->values[i],
                       "compress_matrix: compressed_matrix->values[i] realloc");
      compressed_matrix->col_indices[i] = (int *)realloc(
          compressed_matrix->col_indices[i], non_zeroes_in_row * sizeof(int));
      check_allocation(
          compressed_matrix->col_indices[i],
          "compress_matrix: compressed_matrix->col_indices[i] realloc");
    }

    compressed_matrix->row_sizes[i] =
        non_zeroes_in_row; // Track non-zero elements in the row
  }
  return compressed_matrix;
}

void write_compressed_matrix_to_file(FILE *fileB, FILE *fileC,
                                     CompressedMatrix *compressed_matrix) {
  if (compressed_matrix == NULL) {
    fprintf(stderr, "Compressed Matrix is NULL.\n");
    exit(EXIT_FAILURE);
  }
  if (compressed_matrix->values == NULL ||
      compressed_matrix->col_indices == NULL ||
      compressed_matrix->row_sizes == NULL) {
    fprintf(stderr, "Compressed matrix data is NULL.\n");
    exit(EXIT_FAILURE);
  }

  // Values to fileB
  fprintf(fileB, "%s - Values. Rows: %d\n", compressed_matrix->name,
          compressed_matrix->rows);
  for (int i = 0; i < compressed_matrix->rows; i++) {
    for (int j = 0; j < compressed_matrix->row_sizes[i]; j++) {
      fprintf(fileB, "%3d ", compressed_matrix->values[i][j]);
    }
    fprintf(fileB, "\n");
  }

  // Column indices to fileC
  fprintf(fileC, "%s - Column indices. Rows: %d\n", compressed_matrix->name,
          compressed_matrix->rows);
  for (int i = 0; i < compressed_matrix->rows; i++) {
    for (int j = 0; j < compressed_matrix->row_sizes[i]; j++) {
      fprintf(fileC, "%3d ", compressed_matrix->col_indices[i][j]);
    }
    fprintf(fileC, "\n");
  }
}

void write_uncompressed_matrix_to_file(FILE *file, Matrix *matrix) {
  if (matrix == NULL) {
    fprintf(stderr, "Matrix is NULL.\n");
    exit(EXIT_FAILURE);
  }
  if (matrix->data == NULL) {
    fprintf(stderr, "Matrix data is NULL.\n");
    exit(EXIT_FAILURE);
  }

  fprintf(file, "%s\n", matrix->name);
  fprintf(file, "Rows: %d Cols: %d\n", matrix->rows, matrix->cols);
  for (int i = 0; i < matrix->rows; i++) {
    for (int j = 0; j < matrix->cols; j++) {
      fprintf(file, "%3d ", matrix->data[i][j]);
    }
    fprintf(file, "\n");
  }
}

int get_serialised_size(CompressedMatrix *matrix) {
  // The size includes:
  // 1. `rows` elements for row_sizes
  // 2. Total number of non-zero elements for values and col_indices
  int total_non_zero = 0;
  for (int i = 0; i < matrix->rows; i++) {
    total_non_zero += matrix->row_sizes[i];
  }
  // The total serialized size is row_sizes + values + col_indices + 1 for the
  // total_size
  return matrix->rows + 2 * total_non_zero;
}

/**
 * Serialise a compressed matrix into a 1D array for MPI communication.
 * The serialised array will have the following format:
 *
 * [row_sizes[0], value[0][0], col_indices[0][0], value[0][1],
 * col_indices[0][1], ..., row_sizes[1], value[1][0], col_indices[1][0],
 * value[1][1], col_indices[1][1], ..., row_sizes[2], value[2][0],
 * col_indices[2][0], value[2][1], col_indices[2][1], ...,
 * ...
 * row_sizes[rows-1], value[rows-1][0], col_indices[rows-1][0],
 * value[rows-1][1], col_indices[rows-1][1], ...]
 *
 * The first element of each row is the number of non-zero elements in that row.
 * The following elements are the non-zero values and their corresponding column
 * indices.
 *
 * @param compressed Compressed matrix to serialise
 * @return Serialised array
 */
int *serialise_compressed_matrix(CompressedMatrix *compressed) {
  int total_size = get_serialised_size(compressed);
  int *serialised_matrix = (int *)malloc(total_size * sizeof(int));
  int index = 0;
  for (int i = 0; i < compressed->rows; i++) {
    serialised_matrix[index++] = compressed->row_sizes[i];
    for (int j = 0; j < compressed->row_sizes[i]; j++) {
      serialised_matrix[index++] = compressed->values[i][j];
      serialised_matrix[index++] = compressed->col_indices[i][j];
    }
  }
  return serialised_matrix;
}

CompressedMatrix *deserialise_compressed_matrix(int *serialised, int rows,
                                                int cols) {
  CompressedMatrix *compressed = initialise_compressed_matrix(rows, cols);

  int serialised_idx = 0;
  for (int current_row = 0; current_row < rows; current_row++) {
    // Step 1: Deserialize row_size
    compressed->row_sizes[current_row] = serialised[serialised_idx++];
    // Step 2: Allocate memory for values and col_indices based on row_size
    compressed->values[current_row] =
        (int *)realloc(compressed->values[current_row],
                       compressed->row_sizes[current_row] * sizeof(int));
    compressed->col_indices[current_row] =
        (int *)realloc(compressed->col_indices[current_row],
                       compressed->row_sizes[current_row] * sizeof(int));
    // Step 3: Deserialize values and col_indices
    for (int i = 0; i < compressed->row_sizes[current_row]; i++) {
      compressed->values[current_row][i] = serialised[serialised_idx++];
      compressed->col_indices[current_row][i] = serialised[serialised_idx++];
    }
  }

  return compressed;
}

int *serialise_uncompressed_matrix(Matrix *matrix) {
  int *serialised_matrix =
      (int *)malloc(matrix->rows * matrix->cols * sizeof(int));
  int index = 0;
  for (int i = 0; i < matrix->rows; i++) {
    for (int j = 0; j < matrix->cols; j++) {
      serialised_matrix[index++] = matrix->data[i][j];
    }
  }
  return serialised_matrix;
}

Matrix *deserialise_uncompressed_matrix(int *serialised, int rows, int cols) {
  Matrix *matrix = initialise_matrix(rows, cols, NULL);
  int serialised_idx = 0;
  for (int i = 0; i < matrix->rows; i++) {
    for (int j = 0; j < matrix->cols; j++) {
      matrix->data[i][j] = serialised[serialised_idx++];
    }
  }
  return matrix;
}

Matrix *multiply_compressed_matrices(CompressedMatrix *compressed_matrix_X,
                                     CompressedMatrix *compressed_matrix_Y) {
  if (compressed_matrix_X->max_cols != compressed_matrix_Y->rows) {
    fprintf(stderr,
            "Matrix dimensions are not compatible for multiplication.\n");
    exit(EXIT_FAILURE);
  }

  double start_time, end_time;

  start_time = omp_get_wtime();
  Matrix *result = initialise_matrix(compressed_matrix_X->rows,
                                     compressed_matrix_Y->max_cols, NULL);

#pragma omp parallel for default(shared) schedule(runtime)
  for (int i = 0; i < compressed_matrix_X->rows; i++) {
    // For each row in compressed X
    for (int k = 0; k < compressed_matrix_X->row_sizes[i]; k++) {
      int col_X = compressed_matrix_X->col_indices[i][k];
      int value_X =
          compressed_matrix_X->values[i][k]; // Value from X at (i, col_X)

      // Check elements in `col_X`-th row of Y
      for (int l = 0; l < compressed_matrix_Y->row_sizes[col_X]; l++) {
        int col_Y = compressed_matrix_Y->col_indices[col_X][l];
        int value_Y = compressed_matrix_Y
                          ->values[col_X][l]; // Value from Y at (col_X, col_Y)

        // Update the dot product at position (i, col_Y) in the result matrix
        result->data[i][col_Y] += value_X * value_Y;
      }
    }
  }
  end_time = omp_get_wtime();
  printf("Compressed multiplication time: %fs\n", end_time - start_time);

  return result;
}

Matrix *multiply_uncompressed_matrices(Matrix *matrix_X, Matrix *matrix_Y) {
  if (matrix_X->cols != matrix_Y->rows) {
    fprintf(stderr,
            "Matrix dimensions are not compatible for multiplication.\n");
    exit(EXIT_FAILURE);
  }

  double start_time, end_time;

  start_time = omp_get_wtime();
  Matrix *matrix_result =
      initialise_matrix(matrix_X->rows, matrix_Y->cols, NULL);

#pragma omp parallel for default(shared) schedule(runtime) collapse(2)
  for (int i = 0; i < matrix_X->rows; i++) {
    for (int j = 0; j < matrix_Y->cols; j++) {
      int sum = 0;
      for (int k = 0; k < matrix_X->cols; k++) {
        sum += matrix_X->data[i][k] * matrix_Y->data[k][j];
      }
      matrix_result->data[i][j] = sum;
    }
  }
  end_time = omp_get_wtime();
  printf("Uncompressed multiplication time: %fs\n", end_time - start_time);

  return matrix_result;
}

CompressedMatrix *
gather_partial_compressed_matrices(CompressedMatrix *partial_compressed_Y,
                                   int size, int num_processes, int rank) {
  // Serialise partial_compressed_Y
  int *serialised_partial_compressed_Y =
      serialise_compressed_matrix(partial_compressed_Y);
  int serialised_Y_size = get_serialised_size(partial_compressed_Y);

  // All process gather serialised_Y_sizes to populate recvcounts
  int *recvcounts = (int *)malloc(num_processes * sizeof(int));
  MPI_Allgather(&serialised_Y_size, 1, MPI_INT, recvcounts, 1, MPI_INT,
                MPI_COMM_WORLD);

  // Set displs based on recvcounts
  int *displs = (int *)malloc(num_processes * sizeof(int));
  displs[0] = 0;
  for (int i = 1; i < num_processes; i++) {
    displs[i] = displs[i - 1] + recvcounts[i - 1];
  }

  // All process gather serialised_partial_compressed_Y into
  // serialised_complete_compressed_Y
  int total_serialised_Y_size =
      displs[num_processes - 1] + recvcounts[num_processes - 1];
  int *serialised_complete_compressed_Y =
      (int *)malloc(total_serialised_Y_size * sizeof(int));
  MPI_Allgatherv(serialised_partial_compressed_Y, serialised_Y_size, MPI_INT,
                 serialised_complete_compressed_Y, recvcounts, displs, MPI_INT,
                 MPI_COMM_WORLD);

  // Deserialise serialised_complete_compressed_Y. Now all processes have the
  // complete compressed matrix Y.
  CompressedMatrix *complete_compressed_Y = deserialise_compressed_matrix(
      serialised_complete_compressed_Y, size, size);

  return complete_compressed_Y;
}

Matrix *
gather_partial_uncompressed_matrices(Matrix *partial_uncompressed_result,
                                     int size, int num_processes, int rank) {
  int rows_per_proc = size / num_processes;
  int *serialized_partial_result =
      serialise_uncompressed_matrix(partial_uncompressed_result);
  int serialized_size = rows_per_proc * size;
  int *serialized_complete_result = (int *)malloc(size * size * sizeof(int));

  MPI_Allgather(serialized_partial_result, serialized_size, MPI_INT,
                serialized_complete_result, serialized_size, MPI_INT,
                MPI_COMM_WORLD);

  Matrix *complete_uncompressed_result =
      deserialise_uncompressed_matrix(serialized_complete_result, size, size);

  return complete_uncompressed_result;
}

int compare_uncompressed_matrices(Matrix *matrix1, Matrix *matrix2) {
  if (matrix1->rows != matrix2->rows || matrix1->cols != matrix2->cols) {
    fprintf(stderr, "Matrices have different dimensions.\n");
    return 0;
  }

  for (int i = 0; i < matrix1->rows; i++) {
    for (int j = 0; j < matrix1->cols; j++) {
      if (matrix1->data[i][j] != matrix2->data[i][j]) {
        fprintf(stderr, "Matrices not equal at (%d, %d): %d != %d\n", i, j,
                matrix1->data[i][j], matrix2->data[i][j]);
        return 0;
      }
    }
  }
  return 1;
}

void free_matrix(Matrix *matrix) {
  for (int i = 0; i < matrix->rows; i++) {
    free(matrix->data[i]);
  }
  free(matrix->data);
}

void free_compressed_matrix(CompressedMatrix *matrix) {
  for (int i = 0; i < matrix->rows; i++) {
    free(matrix->values[i]);
    free(matrix->col_indices[i]);
  }
  free(matrix->values);
  free(matrix->col_indices);
  free(matrix->row_sizes);
  free(matrix);
}