/* Replacement for AN5D-Artifact/common.h used by bench/counterparts.py.
 *
 * The AN5D generated host code (<stencil>_host.cu) includes "common.h" and calls back into
 * SB_START_INSTRUMENTS / SB_STOP_INSTRUMENTS around its kernel launches (device synchronised,
 * host and device copies excluded). This header keeps that timing contract unchanged and only
 * swaps the driver: the initial grid can be read from a raw float32 file, the final grid is
 * written back for verification, and timings are printed one per line for the Python driver.
 *
 *   a.out -s <interior size> -t <steps> -n <repeats> [-w <warm-up steps>] [-m <min ms>]
 *         [-i in.f32] [-o out.f32]
 *
 * With -m the warm-up run (after a 1-step run that loads the module) calibrates the number of
 * steps so that every timed repeat lasts at least <min ms>.
 */
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define STENCILBENCH

#define SB_START_INSTRUMENTS do { start_time = sb_time(); } while (0)
#define SB_STOP_INSTRUMENTS  do { end_time   = sb_time(); } while (0)

#ifndef SB_TYPE
#define SB_TYPE float
#endif

static double sb_time()
{
  struct timespec tp;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &tp) != 0)
    perror("clock_gettime failed");
  return (tp.tv_sec + tp.tv_nsec * 1.0e-9) * 1.0e3;
}

double kernel_stencil(SB_TYPE *A1, int compsize, int timestep, bool scop);

int main(int argc, char **argv)
{
  int compsize = 0, timestep = 1, reptnum = 1, warmup = 0, opt;
  double min_ms = 0.0;
  const char *infile = NULL, *outfile = NULL;
  while ((opt = getopt(argc, argv, "s:t:n:w:m:i:o:")) != -1) {
    switch (opt) {
    case 's': compsize = (int)strtol(optarg, NULL, 10); break;
    case 't': timestep = (int)strtol(optarg, NULL, 10); break;
    case 'n': reptnum = (int)strtol(optarg, NULL, 10); break;
    case 'w': warmup = (int)strtol(optarg, NULL, 10); break;
    case 'm': min_ms = strtod(optarg, NULL); break;
    case 'i': infile = optarg; break;
    case 'o': outfile = optarg; break;
    default: return EXIT_FAILURE;
    }
  }
  if (compsize < 1 || timestep < 1 || reptnum < 1) {
    fprintf(stderr, "need -s >= 1, -t >= 1, -n >= 1\n");
    return EXIT_FAILURE;
  }
  size_t dimsize = (size_t)compsize + 2 * BENCH_RAD, gridarea = 1;
  for (int i = 0; i < BENCH_DIM; i++)
    gridarea *= dimsize;
  SB_TYPE *A = (SB_TYPE *)malloc(2 * gridarea * sizeof(SB_TYPE));
  if (A == NULL) {
    fprintf(stderr, "host allocation failed\n");
    return EXIT_FAILURE;
  }
  if (infile) {
    FILE *f = fopen(infile, "rb");
    if (!f || fread(A, sizeof(SB_TYPE), gridarea, f) != gridarea) {
      fprintf(stderr, "cannot read %s\n", infile);
      return EXIT_FAILURE;
    }
    fclose(f);
  } else {
    for (size_t i = 0; i < gridarea; i++) {
      unsigned int seed = (unsigned int)(i * 2654435761u + 12345u);
      A[i] = (SB_TYPE)rand_r(&seed) / (SB_TYPE)RAND_MAX;
    }
  }
  /* both ping-pong buffers carry the same values, so Dirichlet boundaries match */
  memcpy(A + gridarea, A, gridarea * sizeof(SB_TYPE));

  if (warmup > 0) {
    printf("load_ms %.6f\n", kernel_stencil(A, compsize, 1, true));
    double ms = kernel_stencil(A, compsize, warmup, true);
    printf("warmup_ms %.6f\n", ms);
    if (min_ms > 0.0) {
      int need = (int)ceil(min_ms / (ms / warmup));
      timestep = need > timestep ? need : timestep;
    }
  }
  printf("steps %d\n", timestep);
  for (int n = 0; n < reptnum; n++)
    printf("run_ms %.6f\n", kernel_stencil(A, compsize, timestep, true));

  if (outfile) {
    FILE *f = fopen(outfile, "wb");
    if (!f || fwrite(A + (size_t)(timestep % 2) * gridarea, sizeof(SB_TYPE), gridarea, f)
                  != gridarea) {
      fprintf(stderr, "cannot write %s\n", outfile);
      return EXIT_FAILURE;
    }
    fclose(f);
  }
  free(A);
  return EXIT_SUCCESS;
}
