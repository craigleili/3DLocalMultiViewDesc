#include "utils.cuh"

namespace { // kernel namespace

template <typename scalar_t=float>
__device__ __forceinline__ scalar_t sigmoid(const scalar_t x) {
    return 1. / (1. + exp(-x));
}

template <typename scalar_t=float>
__device__ __forceinline__ scalar_t distance_p2p(const scalar_t x1, const scalar_t y1, const scalar_t x2, const scalar_t y2) {
    const scalar_t dx = x1 - x2;
    const scalar_t dy = y1 - y2;
    return sqrt(dx * dx + dy * dy);
}

template <typename scalar_t=float>
__device__ __forceinline__ scalar_t distance2_p2p(const scalar_t x1, const scalar_t y1, const scalar_t x2, const scalar_t y2) {
    const scalar_t dx = x1 - x2;
    const scalar_t dy = y1 - y2;
    return dx * dx + dy * dy;
}

template <typename scalar_t=float>
__device__ __forceinline__ void world_to_dc(const scalar_t* mvp, const scalar_t* xyz1, scalar_t* dc) {
    for (int r = 0; r < 4; ++r) {
        dc[r] = 0.;
        for (int c = 0; c < 4; ++c) {
            dc[r] += mvp[r * 4 + c] * xyz1[c];
        }
    }
}

template <typename scalar_t=float>
__device__ __forceinline__ void dc_to_ndc(const scalar_t* dc, scalar_t* ndc) {
    for (int r = 0; r < 3; ++r) {
        ndc[r] = (dc[r] / dc[3] + 1.) * 0.5;
    }
}

template <typename scalar_t=float, typename index_t=int>
__global__ void soft_rasterize_forward_cuda_kernel(
    // Inputs
    const scalar_t* __restrict__ mvps,
    const scalar_t* __restrict__ vertices,
    const scalar_t* __restrict__ radii,
    const scalar_t* __restrict__ colors,
    index_t*        __restrict__ locks,
    const scalar_t               sigma,
    const scalar_t               gamma,
    const scalar_t               dist_ratio,
    const scalar_t               znear,
    const scalar_t               zfar,
    const scalar_t               tan_half_fov,
    const index_t                num_points,
    const index_t                image_size,
    const index_t                loops,
    const bool                   compute_weight,
    const bool                   draw_color,
    const bool                   draw_depth,
    // Outputs
    scalar_t*       __restrict__ weights,
    scalar_t*       __restrict__ color_map,
    scalar_t*       __restrict__ depth_map,
    scalar_t*       __restrict__ pseudo_depth_map) {

    const index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= loops) return;

    const index_t image_id = i / num_points;
    const index_t image_id3 = image_id * 3;
    const index_t point_id = i % num_points;
    const index_t point_id3 = point_id * 3;

    const scalar_t epsilon = 1e-5;

    const scalar_t* mvp = &mvps[image_id * 16]; 
    scalar_t point_world[4] = {vertices[point_id3], vertices[point_id3 + 1], vertices[point_id3 + 2], 1.}; 

    scalar_t point_dc[4] = {0., 0., 0., 0.};
    scalar_t point[3] = {0., 0., 0.};
    world_to_dc(mvp, point_world, point_dc);
    if (abs(point_dc[3]) < epsilon) return;
    dc_to_ndc(point_dc, point);

    if (point[0] < 0 || point[0] > 1 || point[1] < 0 || point[1] > 1 || point[2] < 0 || point[2] > 1) return;

    const index_t image_size2 = image_size * image_size;
    const scalar_t depth_cam = 2.0 * znear * zfar / (zfar + znear - (2. * point[2] - 1.) * (zfar - znear)); 
    if (radii[point_id] < epsilon) return;
    const scalar_t radius = radii[point_id] / (tan_half_fov * depth_cam);
    scalar_t dist_thresh = radius;
    if (compute_weight) {
        dist_thresh *= dist_ratio;
    }

    const index_t px_min = max(floor((point[0] - dist_thresh) * image_size), 0.);
    const index_t px_max = min(ceil((point[0] + dist_thresh) * image_size), image_size - 1.);
    const index_t py_min = max(floor((point[1] - dist_thresh) * image_size), 0.);
    const index_t py_max = min(ceil((point[1] + dist_thresh) * image_size), image_size - 1.);

    for (index_t px = px_min; px <= px_max; ++px) {
        const scalar_t pxf = (scalar_t) px / (image_size - 1.);
        for (index_t py = py_min; py <= py_max; ++py) {
            const index_t pid = (image_size - 1 - py) * image_size + px; 
            const scalar_t pyf = (scalar_t) py / (image_size - 1.);

            const scalar_t dist2 = distance2_p2p(point[0], point[1], pxf, pyf);
            const scalar_t dist = sqrt(dist2);
            
            const index_t gpid = image_id * image_size2 + pid;
            if (compute_weight) {
                if (dist > dist_thresh) continue;
                const scalar_t dist2_diff = dist2 - radius * radius;
                const scalar_t sign = dist2_diff > 0 ? -1 : 1;
                const scalar_t prob = sigmoid(sign * dist2_diff / (sigma * sigma));
                const scalar_t wtop = prob * exp(-point[2] * gamma);
                atomicAdd(&weights[gpid], wtop);
            }
            
            if (dist > radius) continue;
            index_t locked = 0;
            do {
                if ((locked = atomicCAS(&locks[gpid], 0, 1)) == 0) {
                    if (atomicAdd(&pseudo_depth_map[gpid], 0.) > point[2]) {
                        atomicExch(&pseudo_depth_map[gpid], point[2]);
                        if (draw_color) {
                            const scalar_t* color = &colors[point_id3]; 
                            for (int k = 0; k < 3; ++k) {
                                atomicExch(&color_map[(image_id3 + k) * image_size2 + pid], color[k]);
                            }
                        }
                        if (draw_depth) {
                            atomicExch(&depth_map[gpid], depth_cam);
                        }
                    }
                    atomicExch(&locks[gpid], 0);
                }
            } while(locked > 0);
        }
    }
}

template <typename scalar_t=float, typename index_t=int>
__global__ void soft_rasterize_backward_cuda_kernel(
    // Inputs
    const scalar_t* __restrict__ mvps,
    const scalar_t* __restrict__ vertices,
    const scalar_t* __restrict__ radii,
    const scalar_t* __restrict__ colors,
    const scalar_t* __restrict__ weights,
    const scalar_t* __restrict__ grad_color_map,
    const scalar_t* __restrict__ grad_depth_map,
    const scalar_t               sigma,
    const scalar_t               gamma,
    const scalar_t               dist_ratio,
    const scalar_t               znear,
    const scalar_t               zfar,
    const scalar_t               tan_half_fov,
    const index_t                num_points,
    const index_t                image_size,
    const index_t                loops,
    const bool                   draw_color,
    const bool                   draw_depth,
    // Outputs
    scalar_t*       __restrict__ grad_mvps) {

    const index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= loops) return;

    const index_t image_id = i / num_points;
    const index_t image_id3 = image_id * 3;
    const index_t point_id = i % num_points;
    const index_t point_id3 = point_id * 3;

    const scalar_t epsilon = 1e-5;

    const scalar_t* mvp = &mvps[image_id * 16]; 
    scalar_t point_world[4] = {vertices[point_id3], vertices[point_id3 + 1], vertices[point_id3 + 2], 1.}; 

    scalar_t point_dc[4] = {0., 0., 0., 0.};
    scalar_t point[3] = {0., 0., 0.};
    world_to_dc(mvp, point_world, point_dc);
    if (abs(point_dc[3]) < epsilon) return;
    dc_to_ndc(point_dc, point);

    if (point[0] < 0 || point[0] > 1 || point[1] < 0 || point[1] > 1 || point[2] < 0 || point[2] > 1) return;

    const index_t image_size2 = image_size * image_size;
    const scalar_t depth_cam = 2.0 * znear * zfar / (zfar + znear - (2. * point[2] - 1.) * (zfar - znear)); 
    if (radii[point_id] < epsilon) return; 
    const scalar_t radius = radii[point_id] / (tan_half_fov * depth_cam);
    const scalar_t dist_thresh = dist_ratio * radius; 

    const index_t px_min = max(floor((point[0] - dist_thresh) * image_size), 0.);
    const index_t px_max = min(ceil((point[0] + dist_thresh) * image_size), image_size - 1.);
    const index_t py_min = max(floor((point[1] - dist_thresh) * image_size), 0.);
    const index_t py_max = min(ceil((point[1] + dist_thresh) * image_size), image_size - 1.);
 
    const scalar_t d_dc_z_deno = znear * point[2] - zfar * (point[2] - 1);
    const scalar_t d_dc_z = zfar * znear * (zfar - znear) / (d_dc_z_deno * d_dc_z_deno);
    const scalar_t d_r_z = -(radii[point_id] / tan_half_fov) / (depth_cam * depth_cam) * d_dc_z;
    const scalar_t d_d2d_z = -2 * radius * d_r_z;

    scalar_t grad_point[3] = {0, 0, 0};
    for (index_t px = px_min; px <= px_max; ++px) {
        const scalar_t pxf = (scalar_t) px / (image_size - 1.);
        for (index_t py = py_min; py <= py_max; ++py) {
            const index_t pid = (image_size - 1 - py) * image_size + px; 
            const scalar_t pyf = (scalar_t) py / (image_size - 1.);
            
            const scalar_t dist2 = distance2_p2p(point[0], point[1], pxf, pyf);
            if (sqrt(dist2) > dist_thresh) continue; 

            const scalar_t dist2_diff = dist2 - radius * radius;
            const scalar_t sign = dist2_diff > 0 ? -1 : 1;
            const scalar_t sis = sign / (sigma * sigma);
            const scalar_t prob = sigmoid(sis * dist2_diff);
            const scalar_t ezg = exp(-point[2] * gamma);
            const scalar_t wtop = prob * ezg;
            const scalar_t wsum = weights[image_id * image_size2 + pid];
            const scalar_t pps = prob * (1 - prob) * sis;

            const scalar_t d_wtopsum_wtop = (wsum - wtop) / (wsum * wsum);

            const scalar_t d_prob_z = pps * d_d2d_z;
            const scalar_t d_ezg_z = -gamma * ezg;
            const scalar_t d_wtop_z = prob * d_ezg_z + ezg * d_prob_z;

            const scalar_t d_d2d_x = 2 * (point[0] - pxf);
            const scalar_t d_prob_x = pps * d_d2d_x;
            const scalar_t d_wtop_x = ezg * d_prob_x;

            const scalar_t d_d2d_y = 2 * (point[1] - pyf);
            const scalar_t d_prob_y = pps * d_d2d_y;
            const scalar_t d_wtop_y = ezg * d_prob_y;

            if (draw_color) {
                const scalar_t* color = &colors[point_id3]; 
                for (int k = 0; k < 3; ++k) {
                    const scalar_t d_l_c = grad_color_map[(image_id3 + k) * image_size2 + pid];
                    const scalar_t dcd = d_l_c * color[k] * d_wtopsum_wtop;
                    grad_point[0] += dcd * d_wtop_x;
                    grad_point[1] += dcd * d_wtop_y;
                    grad_point[2] += dcd * d_wtop_z;
                }
            }
            if (draw_depth) {
                const scalar_t dcd = depth_cam * d_wtopsum_wtop;
                const scalar_t d_d_x = dcd * d_wtop_x;
                const scalar_t d_d_y = dcd * d_wtop_y;
                const scalar_t d_d_z = dcd * d_wtop_z + (wtop / wsum) * d_dc_z;

                const scalar_t d_l_d = grad_depth_map[image_id * image_size2 + pid];
                grad_point[0] += d_l_d * d_d_x;
                grad_point[1] += d_l_d * d_d_y;
                grad_point[2] += d_l_d * d_d_z;
            }
        }
    }

    scalar_t* grad_mvp = &grad_mvps[image_id * 16]; 
    for (int r = 0; r < 3; ++r) {
        const scalar_t tmp = grad_point[r] * 0.5 / point_dc[3];
        for (int c = 0; c < 4; ++c) {
            atomicAdd(&grad_mvp[r * 4 + c], tmp * point_world[c]);
        }
    }
    for (int c = 0; c < 4; ++c) {
        const scalar_t tmp = -0.5 * point_world[c] / (point_dc[3] * point_dc[3]);
        scalar_t total = 0;
        for (int k = 0; k < 3; ++k) {
            total += grad_point[k] * tmp * point_dc[k];
        }
        atomicAdd(&grad_mvp[12 + c], total);
    }
}

} // kernel namespace


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
    torch::Tensor pseudo_depth_map) {
    
    if (mvps.dim() != 3) {
        fprintf(stderr,"\nSize of mvps is incorrect.\n");
        exit(-1);
    }
    if (vertices.dim() != 2) {
        fprintf(stderr,"\nSize of vertices is incorrect.\n");
        exit(-1);
    }
    if (compute_weight && weights.dim() != 4){
        fprintf(stderr,"\nSize of weights is incorrect.\n");
        exit(-1);
    }

    const auto num_image = mvps.size(0);
    const auto num_points = vertices.size(0);

    const int loops = num_image * num_points;
    const int threads = MAX_THREADS;
    const int blocks  = gpu_blocks(loops, threads);

    soft_rasterize_forward_cuda_kernel<float, int><<<blocks, threads>>>(
        mvps.data<float>(),
        vertices.data<float>(),
        radii.data<float>(),
        colors.data<float>(),
        locks.data<int>(),
        sigma,
        gamma,
        dist_ratio,
        znear,
        zfar,
        tan_half_fov,
        num_points,
        image_size,
        loops,
        compute_weight,
        draw_color,
        draw_depth,
        weights.data<float>(),
        color_map.data<float>(),
        depth_map.data<float>(),
        pseudo_depth_map.data<float>());
    GPU_ERROR_CHECK(cudaPeekAtLastError());
    GPU_ERROR_CHECK(cudaDeviceSynchronize());
}

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
    torch::Tensor grad_mvps) {

    if (mvps.dim() != 3) {
        fprintf(stderr,"\nSize of mvps is incorrect.\n");
        exit(-1);
    }
    if (vertices.dim() != 2) {
        fprintf(stderr,"\nSize of vertices is incorrect.\n");
        exit(-1);
    }
    if (weights.dim() != 4){
        fprintf(stderr,"\nSize of weights is incorrect.\n");
        exit(-1);
    }

    const auto num_image = mvps.size(0);
    const auto num_points = vertices.size(0);

    const int loops = num_image * num_points;
    const int threads = MAX_THREADS;
    const int blocks  = gpu_blocks(loops, threads);

    soft_rasterize_backward_cuda_kernel<float, int><<<blocks, threads>>>(
        mvps.data<float>(),
        vertices.data<float>(),
        radii.data<float>(),
        colors.data<float>(),
        weights.data<float>(),
        grad_color_map.data<float>(),
        grad_depth_map.data<float>(),
        sigma,
        gamma,
        dist_ratio,
        znear,
        zfar,
        tan_half_fov,
        num_points,
        image_size,
        loops,
        draw_color,
        draw_depth,
        grad_mvps.data<float>());
    GPU_ERROR_CHECK(cudaPeekAtLastError());
    GPU_ERROR_CHECK(cudaDeviceSynchronize());
}
