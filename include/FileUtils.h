// FileUtils.h
#ifndef FILEUTILS_H
#define FILEUTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>

// 函数声明
unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
unsigned char *load_model(const char *filename, int *model_size);
int saveFloat(const char *file_name, float *output, int element_size);

#ifdef __cplusplus
}
#endif

#endif // FILEUTILS_H
