#include <torch/extension.h>

#include <vector>
#include <iostream>

void soft_rasterize_forward_cuda(
    // Inputs
    torch::Tensor mvps,
    torch::Tensor vertices,
    torch::Tensor radii,
    torch::Tensor colors,
    torch::Tensor locks,
    float sigma,
    float gamma,
    float dist_ratio,
    float znear,
    float zfar,
    float tan_half_fov,
    int image_size,
    bool compute_weight,
    bool draw_color,
    bool draw_depth,
    // Outputs
    torch::Tensor weights,
    torch::Tensor color_map,
    torch::Tensor depth_map,
    torch::Tensor pseudo_depth_map);

void soft_rasterize_backward_cuda(
    // Inputs
    torch::Tensor mvps,
    torch::Tensor vertices,
    torch::Tensor radii,
    torch::Tensor colors,
    torch::Tensor weights,
    torch::Tensor grad_color_map,
    torch::Tensor grad_depth_map,
    float sigma,
    float gamma,
    float dist_ratio,
    float znear,
    float zfar,
    float tan_half_fov,
    int image_size,
    bool draw_color,
    bool draw_depth,
    // Outputs
    torch::Tensor grad_mvps);


#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x, " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void soft_rasterize_forward(
    // Inputs
    torch::Tensor mvps,
    torch::Tensor vertices,
    torch::Tensor radii,
    torch::Tensor colors,
    torch::Tensor locks,
    float sigma,
    float gamma,
    float dist_ratio,
    float znear,
    float zfar,
    float tan_half_fov,
    int image_size,
    bool compute_weight,
    bool draw_color,
    bool draw_depth,
    // Outputs
    torch::Tensor weights,
    torch::Tensor color_map,
    torch::Tensor depth_map,
    torch::Tensor pseudo_depth_map) {
    
    CHECK_INPUT(mvps);
    CHECK_INPUT(vertices);
    CHECK_INPUT(radii);
    CHECK_INPUT(colors);
    CHECK_INPUT(locks);
    CHECK_INPUT(weights);
    CHECK_INPUT(color_map);
    CHECK_INPUT(depth_map);
    CHECK_INPUT(pseudo_depth_map);

    return soft_rasterize_forward_cuda(
        mvps,
        vertices,
        radii,
        colors,
        locks, 
        sigma,
        gamma,
        dist_ratio,
        znear,
        zfar,
        tan_half_fov,
        image_size,
        compute_weight,
        draw_color,
        draw_depth,
        weights,
        color_map,
        depth_map,
        pseudo_depth_map);
}

void soft_rasterize_backward(
    // Inputs
    torch::Tensor mvps,
    torch::Tensor vertices,
    torch::Tensor radii,
    torch::Tensor colors,
    torch::Tensor weights,
    torch::Tensor grad_color_map,
    torch::Tensor grad_depth_map,
    float sigma,
    float gamma,
    float dist_ratio,
    float znear,
    float zfar,
    float tan_half_fov,
    int image_size,
    bool draw_color,
    bool draw_depth,
    // Outputs
    torch::Tensor grad_mvps) {
    
    CHECK_INPUT(mvps);
    CHECK_INPUT(vertices);
    CHECK_INPUT(colors);
    CHECK_INPUT(weights);
    CHECK_INPUT(grad_color_map);
    CHECK_INPUT(grad_depth_map);
    CHECK_INPUT(grad_mvps);

    return soft_rasterize_backward_cuda(
        mvps,
        vertices,
        radii,
        colors,
        weights,
        grad_color_map,
        grad_depth_map,
        sigma,
        gamma,
        dist_ratio,
        znear,
        zfar,
        tan_half_fov,
        image_size,
        draw_color,
        draw_depth,
        grad_mvps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("soft_rasterize_forward", &soft_rasterize_forward, "Soft Rasterize - Forward Pass (CUDA)");
    m.def("soft_rasterize_backward", &soft_rasterize_backward, "Soft Rasterize - Backward Pass (CUDA)");
}
