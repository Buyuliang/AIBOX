#ifndef RGA_FUNC_H
#define RGA_FUNC_H

#include <dlfcn.h> 
#include "RgaApi.h"

#ifdef __cplusplus
extern "C" {
#endif

// 函数指针类型定义
typedef int  (*FUNC_RGA_INIT)();  // 初始化 RGA
typedef void (*FUNC_RGA_DEINIT)();  // 释放 RGA 资源
typedef int  (*FUNC_RGA_BLIT)(rga_info_t *, rga_info_t *, rga_info_t *);  // 图像处理函数

// RGA context 结构体，包含 RGA 操作相关的函数指针
typedef struct _rga_context {
    void *rga_handle;        // 动态库句柄
    FUNC_RGA_INIT init_func; // RGA 初始化函数指针
    FUNC_RGA_DEINIT deinit_func; // RGA 释放函数指针
    FUNC_RGA_BLIT blit_func; // RGA 图像处理函数指针
} rga_context;

/**
 * @brief 初始化 RGA 上下文
 * @param rga_ctx RGA context 指针
 * @return 0 表示成功，非 0 表示失败
 */
int RGA_init(rga_context* rga_ctx);

/**
 * @brief 使用 RGA 进行快速图像缩放
 * @param rga_ctx RGA context 指针
 * @param src_fd 源图像文件描述符
 * @param src_w 源图像宽度
 * @param src_h 源图像高度
 * @param dst_phys 目标图像物理地址
 * @param dst_w 目标图像宽度
 * @param dst_h 目标图像高度
 */
void img_resize_fast(rga_context *rga_ctx, int src_fd, int src_w, int src_h, uint64_t dst_phys, int dst_w, int dst_h);

/**
 * @brief 使用 RGA 进行慢速图像缩放（虚拟地址操作）
 * @param rga_ctx RGA context 指针
 * @param src_virt 源图像虚拟地址
 * @param src_w 源图像宽度
 * @param src_h 源图像高度
 * @param dst_virt 目标图像虚拟地址
 * @param dst_w 目标图像宽度
 * @param dst_h 目标图像高度
 */
void img_resize_slow(rga_context *rga_ctx, void *src_virt, int src_w, int src_h, void *dst_virt, int dst_w, int dst_h);

/**
 * @brief 释放 RGA 资源
 * @param rga_ctx RGA context 指针
 * @return 0 表示成功，非 0 表示失败
 */
int RGA_deinit(rga_context* rga_ctx);

#ifdef __cplusplus
}
#endif

#endif // RGA_FUNC_H
