using Test
using LinearAlgebra
using Random
using AstraToolbox

geometry = Parallel2DGeometry(Float32[0.0]; detector_count = 2, detector_width = 1.0f0)
geometry_fbp = Parallel2DGeometry(Float32[0.0, pi / 2]; detector_count = 2, detector_width = 1.0f0)

sart_method = SARTMethod(iterations = 1, projection_order = :custom, projection_order_list = (@view Int64[0, 1, 0][1:2]))

@test sart_method.projection_order_list isa Vector{Int32}
@test sart_method.projection_order_list == Int32[0, 1]
@test_throws ArgumentError SARTMethod(iterations = 1, projection_order = :sequential, projection_order_list = [0, 1])

fbp_method_vec = FBPMethod(custom_filter = (@view Float64[1, 2, 3, 4][2:4]))
fbp_method_mat = FBPMethod(custom_filter = (@view reshape(Float64.(1:6), 2, 3)[:, :]))
fdk_method_vec = FDK3DMethod(custom_filter = (@view Float64[1, 2, 3, 4][2:4]))
fdk_method_mat = FDK3DMethod(custom_filter = (@view reshape(Float64.(1:6), 2, 3)[:, :]))

@test fbp_method_vec.custom_filter isa Vector{Float32}
@test fbp_method_mat.custom_filter isa Matrix{Float32}
@test fdk_method_vec.custom_filter isa Vector{Float32}
@test fdk_method_mat.custom_filter isa Matrix{Float32}
@test fbp_method_vec.custom_filter == Float32[2, 3, 4]
@test fbp_method_mat.custom_filter == reshape(Float32.(1:6), 2, 3)
@test fdk_method_vec.custom_filter == Float32[2, 3, 4]
@test fdk_method_mat.custom_filter == reshape(Float32.(1:6), 2, 3)

options2d = ReconstructionOptions2D(
    reconstruction_mask = (@view reshape(Float64[1, 2, 3, 4], 2, 2)[:, :]),
    sinogram_mask = (@view reshape(Float64[1, 0], 2, 1)[:, :]),
)

@test options2d.reconstruction_mask isa Array{Float32,2}
@test options2d.sinogram_mask isa Array{Float32,2}
@test options2d.reconstruction_mask == reshape(Float32[1, 2, 3, 4], 2, 2)
@test options2d.sinogram_mask == reshape(Float32[1, 0], 2, 1)

options3d = ReconstructionOptions3D(
    reconstruction_mask = (@view reshape(Float64.(1:8), 2, 2, 2)[:, :, :]),
    sinogram_mask = (@view reshape(Float64.(1:12), 2, 3, 2)[:, :, :]),
)

@test options3d.reconstruction_mask isa Array{Float32,3}
@test options3d.sinogram_mask isa Array{Float32,3}
@test options3d.reconstruction_mask == reshape(Float32.(1:8), 2, 2, 2)
@test options3d.sinogram_mask == reshape(Float32.(1:12), 2, 3, 2)

volume = reshape(Float32[1, 2, 3, 4], 2, 2)
sinogram = project(volume, geometry; backend = cpu())

@test vec(sinogram) == Float32[4.0, 6.0]

volume64 = reshape(Float64[1, 2, 3, 4], 2, 2)
sinogram64 = project(volume64, geometry; backend = cpu())

@test vec(sinogram64) == Float32[4.0, 6.0]

volume_view = @view reshape(Float32[0, 1, 2, 3, 4, 0], 3, 2)[1:2, :]
sinogram_view = project(volume_view, geometry; backend = cpu())

@test sinogram_view == project(Matrix{Float32}(volume_view), geometry; backend = cpu())

sinogram_bp64 = reshape(Float64[4, 6], 2, 1)
volume_bp64 = backproject(sinogram_bp64, geometry; volume_size = (2, 2), backend = cpu())

@test volume_bp64 == backproject(Float32.(sinogram_bp64), geometry; volume_size = (2, 2), backend = cpu())

sinogram_bp_view = @view reshape(Float32[4, 6, 0, 0], 2, 2)[:, 1:1]
volume_bp_view = backproject(sinogram_bp_view, geometry; volume_size = (2, 2), backend = cpu())

@test volume_bp_view == backproject(Matrix{Float32}(sinogram_bp_view), geometry; volume_size = (2, 2), backend = cpu())

sinogram_fbp = project(volume, geometry_fbp; backend = cpu())
sinogram_fbp64 = Float64.(sinogram_fbp)
reconstruction64 = reconstruct(sinogram_fbp64, geometry_fbp, FBPMethod(); volume_size = (2, 2), backend = cpu())

@test reconstruction64 == reconstruct(Float32.(sinogram_fbp64), geometry_fbp, FBPMethod(); volume_size = (2, 2), backend = cpu())

sinogram_fbp_view = @view reshape(vcat(vec(sinogram_fbp), Float32[0, 0]), 2, 3)[:, 1:2]
reconstruction_view = reconstruct(sinogram_fbp_view, geometry_fbp, FBPMethod(); volume_size = (2, 2), backend = cpu())

@test reconstruction_view == reconstruct(Matrix{Float32}(sinogram_fbp_view), geometry_fbp, FBPMethod(); volume_size = (2, 2), backend = cpu())

# Test add_noise_to_sino
sino = Float32[1.0 2.0; 3.0 4.0]
sino_noise = add_noise_to_sino(sino, 1000.0; seed=1234)
@test sino_noise isa Array{Float32, 2}
@test size(sino_noise) == size(sino)
@test sino_noise != sino

# Test reproducibility
sino_noise2 = add_noise_to_sino(sino, 1000.0; seed=1234)
@test sino_noise == sino_noise2

# Test in-place
sino_copy = copy(sino)
add_noise_to_sino!(sino_copy, 1000.0; seed=1234)
@test sino_copy == sino_noise

# Test Shepp-Logan phantom generator (2D)
p2d = shepp_logan((64, 64))
@test p2d isa Array{Float32, 2}
@test size(p2d) == (64, 64)
@test all(>=(0.0f0), p2d)
@test any(>(0.0f0), p2d)

p2d_orig = shepp_logan(64; modified=false)
@test p2d_orig isa Array{Float32, 2}
@test size(p2d_orig) == (64, 64)
@test p2d_orig != p2d

buf2d = zeros(Float32, 64, 64)
shepp_logan!(buf2d)
@test buf2d == p2d

p2d_geom = shepp_logan(geometry; volume_size=(64, 64))
@test p2d_geom == p2d

# Test Shepp-Logan phantom generator (3D)
p3d = shepp_logan_3d((32, 32, 32))
@test p3d isa Array{Float32, 3}
@test size(p3d) == (32, 32, 32)
@test all(>=(0.0f0), p3d)
@test any(>(0.0f0), p3d)

p3d_orig = shepp_logan_3d(32; modified=false)
@test p3d_orig isa Array{Float32, 3}
@test size(p3d_orig) == (32, 32, 32)
@test p3d_orig != p3d

buf3d = zeros(Float32, 32, 32, 32)
shepp_logan!(buf3d)
@test buf3d == p3d

geom3d = Parallel3DGeometry(Float32[0.0]; detector_count_u = 2, detector_count_v = 2, detector_spacing_u = 1.0f0, detector_spacing_v = 1.0f0)
p3d_geom = shepp_logan(geom3d; volume_size=(32, 32, 32))
@test p3d_geom == p3d

# Test geometry conversion: geom_2vec
# 2D Parallel
geom_par2d = Parallel2DGeometry(Float32[0.0, Float32(pi/2)]; detector_count = 16, detector_width = 1.5f0)
vec_par2d = geom_2vec(geom_par2d)
@test vec_par2d isa ParallelVec2DGeometry
@test vec_par2d.detector_count == 16
@test size(vec_par2d.vectors) == (6, 2)
# θ = 0: ray = (0, -1), det_center = (0, 0), u = (1.5, 0)
@test vec_par2d.vectors[:, 1] ≈ Float32[0.0, -1.0, 0.0, 0.0, 1.5, 0.0]
# θ = pi/2: ray = (1, 0), det_center = (0, 0), u = (0, 1.5)
@test vec_par2d.vectors[:, 2] ≈ Float32[1.0, 0.0, 0.0, 0.0, 0.0, 1.5]

# 2D FanFlat
geom_fan2d = FanFlat2DGeometry(Float32[0.0]; detector_count = 16, origin_source_distance = 200.0f0, origin_detector_distance = 100.0f0, detector_width = 2.0f0)
vec_fan2d = geom_2vec(geom_fan2d)
@test vec_fan2d isa FanFlatVec2DGeometry
@test vec_fan2d.detector_count == 16
@test size(vec_fan2d.vectors) == (6, 1)
# θ = 0: src = (0, -200), det_center = (0, 100), u = (2.0, 0)
@test vec_fan2d.vectors[:, 1] ≈ Float32[0.0, -200.0, 0.0, 100.0, 2.0, 0.0]

# 3D Parallel
geom_par3d = Parallel3DGeometry(Float32[0.0]; detector_count_u = 16, detector_count_v = 8, detector_spacing_u = 1.2f0, detector_spacing_v = 1.4f0)
vec_par3d = geom_2vec(geom_par3d)
@test vec_par3d isa ParallelVec3DGeometry
@test vec_par3d.detector_count_u == 16
@test vec_par3d.detector_count_v == 8
@test size(vec_par3d.vectors) == (12, 1)
# θ = 0: ray = (0, -1, 0), det_center = (0, 0, 0), u = (1.2, 0, 0), v = (0, 0, 1.4)
@test vec_par3d.vectors[:, 1] ≈ Float32[0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0, 1.4]

# 3D Cone
geom_cone3d = Cone3DGeometry(Float32[0.0], 16, 8, 300.0f0, 150.0f0; detector_spacing_u = 1.1f0, detector_spacing_v = 1.3f0)
vec_cone3d = geom_2vec(geom_cone3d)
@test vec_cone3d isa ConeVec3DGeometry
@test vec_cone3d.detector_count_u == 16
@test vec_cone3d.detector_count_v == 8
@test size(vec_cone3d.vectors) == (12, 1)
# θ = 0: src = (0, -300, 0), det_center = (0, 150, 0), u = (1.1, 0, 0), v = (0, 0, 1.3)
@test vec_cone3d.vectors[:, 1] ≈ Float32[0.0, -300.0, 0.0, 0.0, 150.0, 0.0, 1.1, 0.0, 0.0, 0.0, 0.0, 1.3]

# Pass-through copies
@test geom_2vec(vec_par2d).vectors == vec_par2d.vectors
@test geom_2vec(vec_fan2d).vectors == vec_fan2d.vectors
@test geom_2vec(vec_par3d).vectors == vec_par3d.vectors
@test geom_2vec(vec_cone3d).vectors == vec_cone3d.vectors

# CylConeVec3D pass-through
cyl_vecs = zeros(Float32, 13, 2)
cyl_geom = CylConeVec3DGeometry(cyl_vecs; detector_count_u = 4, detector_count_v = 4)
@test geom_2vec(cyl_geom).vectors == cyl_vecs

# Test postalignment
post_par2d = geom_postalignment(geom_par2d, 2.0f0)
@test post_par2d isa ParallelVec2DGeometry
@test post_par2d.vectors[3:4, 1] ≈ Float32[3.0, 0.0] # shifted by 2 * u = 2 * (1.5, 0)

post_cone3d = geom_postalignment(geom_cone3d, (2.0f0, 3.0f0))
@test post_cone3d isa ConeVec3DGeometry
# det center shifted by 2 * (1.1, 0, 0) + 3 * (0, 0, 1.3) = (2.2, 0, 3.9)
@test post_cone3d.vectors[4:6, 1] ≈ Float32[2.2, 150.0, 3.9]

post_cone3d_single = geom_postalignment(geom_cone3d, 1.5f0)
@test post_cone3d_single.vectors[4:6, 1] ≈ Float32[1.65, 150.0, 0.0]

@test_throws ArgumentError geom_postalignment(cyl_geom, 1.0f0)

# Test 2D CPU projection and backprojection with vector geometries
vol2d_test = shepp_logan((16, 16))
sino_par_cpu = project(vol2d_test, vec_par2d; backend = cpu())
bp_par_cpu = backproject(sino_par_cpu, vec_par2d; volume_size = (16, 16), backend = cpu())
@test size(sino_par_cpu) == (16, 2)
@test size(bp_par_cpu) == (16, 16)
@test all(isfinite, sino_par_cpu)
@test all(isfinite, bp_par_cpu)

# Test 3D geometries and reconstructions (when CUDA is functional and device is detected)
cuda_available = has_cuda() && get_gpu_count() > 0 && try
    test_vol = zeros(Float32, 2, 2)
    test_geom = Parallel2DGeometry(Float32[0.0]; detector_count = 2)
    project(test_vol, test_geom; backend = cuda())
    true
catch
    false
end

if cuda_available
    @info "Running 3D GPU geometry & reconstruction test suite"

    # 3D Cone-beam test
    angles_3d = Float32.(range(0, 2pi, length = 8))
    cone_geom = Cone3DGeometry(angles_3d, 8, 8, 100.0f0, 100.0f0; detector_spacing_u = 1.0f0, detector_spacing_v = 1.0f0)
    vol_3d = shepp_logan_3d((8, 8, 8))

    sino_cone = project(vol_3d, cone_geom; backend = cuda())
    @test size(sino_cone) == (8, 8, 8)
    @test all(isfinite, sino_cone)

    # Equivalence with ConeVec3DGeometry
    cone_vec_geom = geom_2vec(cone_geom)
    sino_cone_vec = project(vol_3d, cone_vec_geom; backend = cuda())
    @test sino_cone ≈ sino_cone_vec rtol = 1e-4

    # Backprojection
    bp_cone = backproject(sino_cone, cone_geom; volume_size = (8, 8, 8), backend = cuda())
    @test size(bp_cone) == (8, 8, 8)
    @test all(isfinite, bp_cone)

    # 3D Reconstructions: FDK, SIRT, CGLS
    rec_fdk = reconstruct(sino_cone, cone_geom, FDK3DMethod(); volume_size = (8, 8, 8), backend = cuda())
    @test size(rec_fdk) == (8, 8, 8)
    @test all(isfinite, rec_fdk)

    rec_fdk_vec = reconstruct(sino_cone_vec, cone_vec_geom, FDK3DMethod(); volume_size = (8, 8, 8), backend = cuda())
    @test rec_fdk ≈ rec_fdk_vec rtol = 1e-3

    rec_sirt = reconstruct(sino_cone, cone_geom, SIRT3DMethod(iterations = 2); volume_size = (8, 8, 8), backend = cuda())
    @test size(rec_sirt) == (8, 8, 8)
    @test all(isfinite, rec_sirt)

    rec_cgls = reconstruct(sino_cone, cone_geom, CGLS3DMethod(iterations = 2); volume_size = (8, 8, 8), backend = cuda())
    @test size(rec_cgls) == (8, 8, 8)
    @test all(isfinite, rec_cgls)

    # 3D Parallel-beam test
    angles_par3d = Float32.(range(0, pi, length = 8))
    par3d_geom = Parallel3DGeometry(angles_par3d; detector_count_u = 8, detector_count_v = 8, detector_spacing_u = 1.0f0, detector_spacing_v = 1.0f0)
    sino_par3d = project(vol_3d, par3d_geom; backend = cuda())
    @test size(sino_par3d) == (8, 8, 8)

    par3d_vec_geom = geom_2vec(par3d_geom)
    sino_par3d_vec = project(vol_3d, par3d_vec_geom; backend = cuda())
    @test sino_par3d ≈ sino_par3d_vec rtol = 1e-4

    bp_par3d = backproject(sino_par3d, par3d_geom; volume_size = (8, 8, 8), backend = cuda())
    @test size(bp_par3d) == (8, 8, 8)

    rec_sirt_par = reconstruct(sino_par3d, par3d_geom, SIRT3DMethod(iterations = 2); volume_size = (8, 8, 8), backend = cuda())
    @test size(rec_sirt_par) == (8, 8, 8)

    rec_cgls_par = reconstruct(sino_par3d, par3d_geom, CGLS3DMethod(iterations = 2); volume_size = (8, 8, 8), backend = cuda())
    @test size(rec_cgls_par) == (8, 8, 8)

    # 3D Cylindrical Cone-beam test
    cyl_n = length(angles_3d)
    cyl_mat = zeros(Float32, 13, cyl_n)
    for i in 1:cyl_n
        θ = angles_3d[i]
        sθ, cθ = sincos(θ)
        cyl_mat[1, i] = sθ * 100.0f0
        cyl_mat[2, i] = -cθ * 100.0f0
        cyl_mat[4, i] = -sθ * 100.0f0
        cyl_mat[5, i] = cθ * 100.0f0
        cyl_mat[7, i] = cθ * 0.01f0
        cyl_mat[8, i] = sθ * 0.01f0
        cyl_mat[12, i] = 1.0f0
        cyl_mat[13, i] = 100.0f0
    end
    cyl_geom_test = CylConeVec3DGeometry(cyl_mat; detector_count_u = 8, detector_count_v = 8)
    sino_cyl = project(vol_3d, cyl_geom_test; backend = cuda())
    @test size(sino_cyl) == (8, 8, 8)
    @test all(isfinite, sino_cyl)

    bp_cyl = backproject(sino_cyl, cyl_geom_test; volume_size = (8, 8, 8), backend = cuda())
    @test size(bp_cyl) == (8, 8, 8)
    @test all(isfinite, bp_cyl)

    # 3D AstraProjectorMap test
    A3d = AstraProjectorMap(cone_geom, (8, 8, 8); backend = cuda())
    @test size(A3d) == (512, 512)
    @test A3d.volume_size == (8, 8, 8)
    @test A3d.sinogram_size == (8, 8, 8)

    sino3d_op = A3d * vol_3d
    @test sino3d_op ≈ sino_cone
    sino3d_op_vec = A3d * vec(vol_3d)
    @test sino3d_op_vec ≈ vec(sino_cone)

    bp3d_op = A3d' * sino_cone
    @test bp3d_op ≈ bp_cone
    bp3d_op_vec = A3d' * vec(sino_cone)
    @test bp3d_op_vec ≈ vec(bp_cone)

    y3d_buf = zeros(Float32, 512)
    mul!(y3d_buf, A3d, vec(vol_3d))
    @test y3d_buf ≈ vec(sino_cone)

    x3d_buf = zeros(Float32, 512)
    mul!(x3d_buf, A3d', vec(sino_cone))
    @test x3d_buf ≈ vec(bp_cone)
else
    @warn "CUDA not available; skipping 3D GPU execution tests"
end

# Test clip_circle and clip_circle! (2D and 3D)
img_circ = ones(Float32, 32, 32)
img_clipped = clip_circle(img_circ)
@test size(img_clipped) == (32, 32)
@test img_clipped[16, 16] == 1.0f0
@test img_clipped[1, 1] == 0.0f0
@test img_clipped[1, 32] == 0.0f0
@test img_clipped[32, 1] == 0.0f0
@test img_clipped[32, 32] == 0.0f0

# In-place 2D
img_copy = copy(img_circ)
clip_circle!(img_copy)
@test img_copy == img_clipped

# 3D cylindrical clip
vol_circ = ones(Float32, 16, 16, 8)
vol_clipped = clip_circle(vol_circ)
@test size(vol_clipped) == (16, 16, 8)
@test vol_clipped[8, 8, 4] == 1.0f0
@test vol_clipped[1, 1, 4] == 0.0f0
@test vol_clipped[16, 16, 4] == 0.0f0

# Test 2D AstraProjectorMap
geom2d_op = Parallel2DGeometry(Float32[0.0, Float32(pi/2)]; detector_count = 8)
vol2d_op = shepp_logan((8, 8))
A2d = AstraProjectorMap(geom2d_op, (8, 8))

@test size(A2d) == (16, 64)
@test A2d.volume_size == (8, 8)
@test A2d.sinogram_size == (8, 2)

# Forward projection
sino2d_op_mat = A2d * vol2d_op
sino2d_op_vec = A2d * vec(vol2d_op)
@test size(sino2d_op_mat) == (8, 2)
@test length(sino2d_op_vec) == 16
@test vec(sino2d_op_mat) == sino2d_op_vec

# In-place mul!
y2d_buf = zeros(Float32, 16)
mul!(y2d_buf, A2d, vec(vol2d_op))
@test y2d_buf == sino2d_op_vec

# 5-arg mul!
y2d_5arg = fill(2.0f0, 16)
mul!(y2d_5arg, A2d, vec(vol2d_op), 2.0f0, 3.0f0)
@test y2d_5arg ≈ 2.0f0 .* sino2d_op_vec .+ 6.0f0

# Backprojection
bp2d_op_mat = A2d' * sino2d_op_mat
bp2d_op_vec = A2d' * sino2d_op_vec
@test size(bp2d_op_mat) == (8, 8)
@test length(bp2d_op_vec) == 64
@test vec(bp2d_op_mat) == bp2d_op_vec

# In-place adjoint mul!
x2d_buf = zeros(Float32, 64)
mul!(x2d_buf, A2d', sino2d_op_vec)
@test x2d_buf == bp2d_op_vec

# 5-arg adjoint mul!
x2d_5arg = fill(1.0f0, 64)
mul!(x2d_5arg, A2d', sino2d_op_vec, 2.0f0, 4.0f0)
@test x2d_5arg ≈ 2.0f0 .* bp2d_op_vec .+ 4.0f0

# Matrix multiplication with multiple vectors
V_multi = hcat(vec(vol2d_op), vec(vol2d_op))
S_multi = A2d * V_multi
@test size(S_multi) == (16, 2)
@test S_multi[:, 1] ≈ sino2d_op_vec
@test S_multi[:, 2] ≈ sino2d_op_vec

BP_multi = A2d' * S_multi
@test size(BP_multi) == (64, 2)
@test BP_multi[:, 1] ≈ bp2d_op_vec
@test BP_multi[:, 2] ≈ bp2d_op_vec

# Element indexing
@test A2d[1, 1] isa Float32
@test_throws BoundsError A2d[0, 1]
@test_throws BoundsError A2d[17, 1]

# ---------------- GPU Management Tests ----------------
@test has_cuda() isa Bool
@test has_feature("cuda") == has_cuda()
@test has_feature("non_existent_feature_12345") == false
@test get_gpu_count() isa Int
if has_cuda() && get_gpu_count() > 0
    @test get_gpu_count() >= 1
    info = get_gpu_info()
    @test info isa String
    @test length(info) > 0
    @test get_gpu_index() isa Int
    @test set_gpu_index(0) == true
    @test get_gpu_index() == 0
else
    @test get_gpu_count() >= 0
end

# ---------------- CPU Projection Kernels (Linear & Strip) Tests ----------------
geom_kernels = Parallel2DGeometry(range(0f0, Float32(pi), length=16); detector_count=16)
vol_test = shepp_logan(16)

sino_line = project(vol_test, geom_kernels; backend=cpu(), projector_kernel=:line)
sino_linear = project(vol_test, geom_kernels; backend=cpu(), projector_kernel=:linear)
sino_strip = project(vol_test, geom_kernels; backend=cpu(), projector_kernel=:strip)

@test size(sino_line) == (16, 16)
@test size(sino_linear) == (16, 16)
@test size(sino_strip) == (16, 16)
@test sum(abs.(sino_line)) > 0
@test sum(abs.(sino_linear)) > 0
@test sum(abs.(sino_strip)) > 0
# Line, linear and strip forward projections should agree within a few percent
@test isapprox(sum(sino_line), sum(sino_linear), rtol=0.05)
@test isapprox(sum(sino_line), sum(sino_strip), rtol=0.05)

bp_line = backproject(sino_line, geom_kernels; volume_size=(16, 16), backend=cpu(), projector_kernel=:line)
bp_linear = backproject(sino_line, geom_kernels; volume_size=(16, 16), backend=cpu(), projector_kernel=:linear)
bp_strip = backproject(sino_line, geom_kernels; volume_size=(16, 16), backend=cpu(), projector_kernel=:strip)

@test size(bp_line) == (16, 16)
@test size(bp_linear) == (16, 16)
@test size(bp_strip) == (16, 16)
@test isapprox(sum(bp_line), sum(bp_linear), rtol=0.05)
@test isapprox(sum(bp_line), sum(bp_strip), rtol=0.05)

# Options structs and validation
opt_proj = ProjectionOptions2D(detector_supersampling=1, projector_kernel=:linear)
@test opt_proj.projector_kernel == :linear
@test_throws ArgumentError ProjectionOptions2D(projector_kernel=:invalid)
@test_throws ArgumentError BackprojectionOptions2D(projector_kernel=:invalid)
@test_throws ArgumentError project(vol_test, geom_kernels; backend=cpu(), projector_kernel=:invalid)
@test_throws ArgumentError backproject(sino_line, geom_kernels; volume_size=(16, 16), backend=cpu(), projector_kernel=:invalid)

# ParallelVec kernels
vecs_test = geom_2vec(geom_kernels)
sino_pv_linear = project(vol_test, vecs_test; backend=cpu(), projector_kernel=:linear)
sino_pv_strip = project(vol_test, vecs_test; backend=cpu(), projector_kernel=:strip)
@test isapprox(sum(sino_pv_linear), sum(sino_linear), rtol=1e-4)
@test isapprox(sum(sino_pv_strip), sum(sino_strip), rtol=1e-4)

# Fanflat strip kernel
geom_ff_test = FanFlat2DGeometry(collect(range(0f0, 2f0*Float32(pi), length=16)); detector_count=16, detector_width=1.0f0, origin_source_distance=100.0f0, origin_detector_distance=50.0f0)
sino_ff_line = project(vol_test, geom_ff_test; backend=cpu(), projector_kernel=:line)
sino_ff_strip = project(vol_test, geom_ff_test; backend=cpu(), projector_kernel=:strip)
@test isapprox(sum(sino_ff_line), sum(sino_ff_strip), rtol=0.05)
@test_throws ArgumentError project(vol_test, geom_ff_test; backend=cpu(), projector_kernel=:linear)

# ---------------- CPU Iterative Reconstruction Tests ----------------
# SIRT on CPU
sirt_m = SIRTMethod(iterations=10, min_constraint=0.0, max_constraint=1.0, projector_kernel=:line)
@test sirt_m.projector_kernel == :line
rec_sirt_cpu = reconstruct(sino_line, geom_kernels, sirt_m; volume_size=(16, 16), backend=cpu())
@test size(rec_sirt_cpu) == (16, 16)
@test all(rec_sirt_cpu .>= -1e-5)
@test all(rec_sirt_cpu .<= 1.0 + 1e-5)

# SIRT with linear kernel on CPU
sirt_m_linear = SIRTMethod(iterations=5, projector_kernel=:linear)
rec_sirt_linear = reconstruct(sino_line, geom_kernels, sirt_m_linear; volume_size=(16, 16), backend=cpu())
@test size(rec_sirt_linear) == (16, 16)

# SART on CPU
sart_seq = SARTMethod(iterations=5, projection_order=:sequential, min_constraint=0.0)
rec_sart_cpu = reconstruct(sino_line, geom_kernels, sart_seq; volume_size=(16, 16), backend=cpu())
@test size(rec_sart_cpu) == (16, 16)
@test all(rec_sart_cpu .>= -1e-5)

# SART with custom order on CPU
order_list = Int32.(0:15)
sart_custom = SARTMethod(iterations=3, projection_order=:custom, projection_order_list=order_list)
rec_sart_custom = reconstruct(sino_line, geom_kernels, sart_custom; volume_size=(16, 16), backend=cpu())
@test size(rec_sart_custom) == (16, 16)

# CGLS on CPU
cgls_m = CGLSMethod(iterations=5)
rec_cgls_cpu = reconstruct(sino_line, geom_kernels, cgls_m; volume_size=(16, 16), backend=cpu())
@test size(rec_cgls_cpu) == (16, 16)

# ART on CPU
art_m = ARTMethod(iterations=160, relaxation=0.5, min_constraint=0.0)
@test art_m.iterations == 160
@test art_m.relaxation == 0.5f0
@test art_m.min_constraint == 0.0f0
@test art_m.projector_kernel == :line
rec_art_cpu = reconstruct(sino_line, geom_kernels, art_m; volume_size=(16, 16), backend=cpu())
@test size(rec_art_cpu) == (16, 16)
@test all(rec_art_cpu .>= -1e-5)

# Fanflat and vector CPU reconstructions
rec_ff_sirt = reconstruct(sino_ff_line, geom_ff_test, SIRTMethod(iterations=3); volume_size=(16, 16), backend=cpu())
@test size(rec_ff_sirt) == (16, 16)
rec_pv_sart = reconstruct(sino_pv_linear, vecs_test, SARTMethod(iterations=3); volume_size=(16, 16), backend=cpu())
@test size(rec_pv_sart) == (16, 16)
vecs_ff_test = geom_2vec(geom_ff_test)
rec_ffv_cgls = reconstruct(sino_ff_line, vecs_ff_test, CGLSMethod(iterations=3); volume_size=(16, 16), backend=cpu())
@test size(rec_ffv_cgls) == (16, 16)

# Error cases for CPU iterative methods
@test_throws ArgumentError reconstruct(sino_line, geom_kernels, SIRTMethod(iterations=1, detector_supersampling=2); volume_size=(16, 16), backend=cpu())
@test_throws ArgumentError reconstruct(sino_line, geom_kernels, SIRTMethod(iterations=1); volume_size=(16, 16), backend=cpu(), residual_norm=Ref(0.0f0))
if has_cuda()
    @test_throws ArgumentError reconstruct(sino_line, geom_kernels, ARTMethod(iterations=10); volume_size=(16, 16), backend=cuda())
end

# ---------------- Distance-Driven Projector Tests ----------------
sino_dd = project(vol_test, geom_kernels; backend=cpu(), projector_kernel=:distance_driven)
@test size(sino_dd) == size(sino_line)
@test isapprox(sum(sino_dd), sum(sino_line), rtol=0.05)

bp_dd = backproject(sino_line, geom_kernels; volume_size=(16, 16), backend=cpu(), projector_kernel=:distance_driven)
@test size(bp_dd) == (16, 16)
@test isapprox(sum(bp_dd), sum(bp_line), rtol=0.05)

opt_dd_proj = ProjectionOptions2D(projector_kernel=:distance_driven)
@test opt_dd_proj.projector_kernel == :distance_driven
opt_dd_bp = BackprojectionOptions2D(projector_kernel=:distance_driven)
@test opt_dd_bp.projector_kernel == :distance_driven

# ParallelVec distance-driven
sino_pv_dd = project(vol_test, vecs_test; backend=cpu(), projector_kernel=:distance_driven)
@test isapprox(sum(sino_pv_dd), sum(sino_dd), rtol=1e-4)
bp_pv_dd = backproject(sino_dd, vecs_test; volume_size=(16, 16), backend=cpu(), projector_kernel=:distance_driven)
@test size(bp_pv_dd) == (16, 16)

# Error: fanflat distance_driven is not supported
@test_throws ArgumentError project(vol_test, geom_ff_test; backend=cpu(), projector_kernel=:distance_driven)
@test_throws ArgumentError backproject(sino_ff_line, geom_ff_test; volume_size=(16, 16), backend=cpu(), projector_kernel=:distance_driven)

# Distance-driven iterative reconstruction
rec_sirt_dd = reconstruct(sino_dd, geom_kernels, SIRTMethod(iterations=5, projector_kernel=:distance_driven); volume_size=(16, 16), backend=cpu())
@test size(rec_sirt_dd) == (16, 16)
rec_sart_dd = reconstruct(sino_dd, geom_kernels, SARTMethod(iterations=3, projector_kernel=:distance_driven); volume_size=(16, 16), backend=cpu())
@test size(rec_sart_dd) == (16, 16)
rec_cgls_dd = reconstruct(sino_dd, geom_kernels, CGLSMethod(iterations=3, projector_kernel=:distance_driven); volume_size=(16, 16), backend=cpu())
@test size(rec_cgls_dd) == (16, 16)
rec_art_dd = reconstruct(sino_dd, geom_kernels, ARTMethod(iterations=16, projector_kernel=:distance_driven); volume_size=(16, 16), backend=cpu())
@test size(rec_art_dd) == (16, 16)

# ---------------- Sparse Matrix Geometry & Projector Tests ----------------
using SparseArrays

det_count = 8
ang_count = 4
v_w = 4
v_h = 4
m_rows = det_count * ang_count
n_cols = v_w * v_h

# Create a random sparse matrix
rng_sparse = sprand(Float32, m_rows, n_cols, 0.25)
geom_sparse = SparseMatrix2DGeometry(rng_sparse; detector_count=det_count, volume_size=(v_w, v_h))
@test geom_sparse isa SparseMatrix2DGeometry
@test geom_sparse isa Sparse2DGeometry
@test geom_sparse.detector_count == det_count
@test geom_sparse.angle_count == ang_count
@test geom_sparse.volume_width == v_w
@test geom_sparse.volume_height == v_h

# Test forward projection matches A * vec(vol)
vol_sp = rand(Float32, v_w, v_h)
sino_sp = project(vol_sp, geom_sparse; backend=cpu())
@test size(sino_sp) == (det_count, ang_count)
expected_sino = reshape(Vector(rng_sparse * vec(vol_sp)), det_count, ang_count)
@test isapprox(sino_sp, expected_sino; atol=1e-6)

# Test backprojection matches A' * vec(sino)
sino_input = rand(Float32, det_count, ang_count)
bp_sp = backproject(sino_input, geom_sparse; backend=cpu())
@test size(bp_sp) == (v_w, v_h)
expected_bp = reshape(Vector(rng_sparse' * vec(sino_input)), v_w, v_h)
@test isapprox(bp_sp, expected_bp; atol=1e-6)

# Test in-place FP & BP
sino_out = zeros(Float32, det_count, ang_count)
project!(sino_out, vol_sp, geom_sparse; backend=cpu())
@test isapprox(sino_out, sino_sp; atol=1e-6)

bp_out = zeros(Float32, v_w, v_h)
backproject!(bp_out, sino_input, geom_sparse; backend=cpu())
@test isapprox(bp_out, bp_sp; atol=1e-6)

# Test iterative solvers with sparse matrix geometry
rec_sirt_sp = reconstruct(sino_sp, geom_sparse, SIRTMethod(iterations=5); backend=cpu())
@test size(rec_sirt_sp) == (v_w, v_h)
rec_sart_sp = reconstruct(sino_sp, geom_sparse, SARTMethod(iterations=3); backend=cpu())
@test size(rec_sart_sp) == (v_w, v_h)
rec_cgls_sp = reconstruct(sino_sp, geom_sparse, CGLSMethod(iterations=3); backend=cpu())
@test size(rec_cgls_sp) == (v_w, v_h)
rec_art_sp = reconstruct(sino_sp, geom_sparse, ARTMethod(iterations=10); backend=cpu())
@test size(rec_art_sp) == (v_w, v_h)

# Test in-place reconstruct! with sparse matrix geometry
vol_rec_inplace = zeros(Float32, v_w, v_h)
reconstruct!(vol_rec_inplace, sino_sp, geom_sparse, SIRTMethod(iterations=5); backend=cpu())
@test isapprox(vol_rec_inplace, rec_sirt_sp; atol=1e-6)

# Test dimension checks
@test_throws ArgumentError SparseMatrix2DGeometry(rng_sparse; detector_count=3, volume_size=(v_w, v_h))
@test_throws ArgumentError SparseMatrix2DGeometry(rng_sparse; detector_count=det_count, volume_size=(5, 5))

# -------------------------------------------------------------------------
# Test VolumeGeometry (create_vol_geom, ROI window, pixel_size, voxel_size)
# -------------------------------------------------------------------------

# 2D VolumeGeometry constructors and accessors
vg2d_default = VolumeGeometry2D(128, 64)
@test vg2d_default isa VolumeGeometry2D
@test vg2d_default isa AbstractVolumeGeometry
@test size(vg2d_default) == (128, 64)
@test vg2d_default.window_min_x == -64.0f0
@test vg2d_default.window_max_x == 64.0f0
@test vg2d_default.window_min_y == -32.0f0
@test vg2d_default.window_max_y == 32.0f0
@test pixel_size(vg2d_default) == (1.0f0, 1.0f0)
@test AstraToolbox.center(vg2d_default) == (0.0f0, 0.0f0)

vg2d_win = VolumeGeometry2D((100, 100); window=(-20.0, 20.0, -10.0, 10.0))
@test size(vg2d_win) == (100, 100)
@test pixel_size(vg2d_win) == (0.4f0, 0.2f0)
@test AstraToolbox.center(vg2d_win) == (0.0f0, 0.0f0)

vg2d_pix = VolumeGeometry2D(100, 100; pixel_size=(0.5, 0.5), center=(5.0, 5.0))
@test vg2d_pix.window_min_x == -20.0f0
@test vg2d_pix.window_max_x == 30.0f0
@test vg2d_pix.window_min_y == -20.0f0
@test vg2d_pix.window_max_y == 30.0f0
@test pixel_size(vg2d_pix) == (0.5f0, 0.5f0)
@test AstraToolbox.center(vg2d_pix) == (5.0f0, 5.0f0)

@test_throws ArgumentError VolumeGeometry2D(100, 100; window=(-10, 10, -10, 10), pixel_size=0.5)
@test_throws ArgumentError VolumeGeometry2D(100, 100; window=(10, -10, -10, 10))

# 3D VolumeGeometry constructors and accessors
vg3d_default = VolumeGeometry3D(64, 64, 32)
@test vg3d_default isa VolumeGeometry3D
@test vg3d_default isa AbstractVolumeGeometry
@test size(vg3d_default) == (64, 64, 32)
@test voxel_size(vg3d_default) == (1.0f0, 1.0f0, 1.0f0)
@test AstraToolbox.center(vg3d_default) == (0.0f0, 0.0f0, 0.0f0)

vg3d_win = VolumeGeometry3D((50, 50, 25); window=(-10, 10, -10, 10, -5, 5))
@test size(vg3d_win) == (50, 50, 25)
@test voxel_size(vg3d_win) == (0.4f0, 0.4f0, 0.4f0)
@test AstraToolbox.center(vg3d_win) == (0.0f0, 0.0f0, 0.0f0)

# create_vol_geom upstream compatibility helper
@test create_vol_geom(64) == VolumeGeometry2D(64, 64)
@test create_vol_geom((64, 64)) == VolumeGeometry2D(64, 64)
@test create_vol_geom(64, 32) == VolumeGeometry2D(64, 32)
@test create_vol_geom(64, 32, -10, 10, -5, 5) == VolumeGeometry2D(64, 32; window=(-10, 10, -5, 5))
@test create_vol_geom(64, 64, 32) == VolumeGeometry3D(64, 64, 32)
@test create_vol_geom((64, 64, 32)) == VolumeGeometry3D(64, 64, 32)
@test create_vol_geom(64, 64, 32, -10, 10, -10, 10, -5, 5) == VolumeGeometry3D(64, 64, 32; window=(-10, 10, -10, 10, -5, 5))

# Physical projection scaling test with custom window
# 4x4 uniform volume, ray along y-axis
geom_par1 = Parallel2DGeometry(Float32[0.0]; detector_count=10, detector_width=1.0f0)
vol4x4 = ones(Float32, 4, 4)

# Default window: min_x=-2, max_x=2, min_y=-2, max_y=2. Pixel height = 1.0. Ray through 4 pixels = 4.0.
sino_def = project(vol4x4, geom_par1; backend=cpu())

# Scaled window: min_x=-4, max_x=4, min_y=-4, max_y=4. Pixel height = 2.0. Ray through 4 pixels = 8.0.
vg_scaled = VolumeGeometry2D(4, 4; window=(-4, 4, -4, 4))
sino_scaled = project(vol4x4, geom_par1, vg_scaled; backend=cpu())
sino_scaled_kw = project(vol4x4, geom_par1; volume_geometry=vg_scaled, backend=cpu())
sino_scaled_win = project(vol4x4, geom_par1; window=(-4, 4, -4, 4), backend=cpu())

@test sino_scaled == sino_scaled_kw
@test sino_scaled == sino_scaled_win
# In the center (detectors 5 and 6), scaled ray length should be exactly 2x default
@test isapprox(sino_scaled[5, 1], 2.0f0 * sino_def[5, 1]; atol=1e-5)
@test isapprox(sino_scaled[6, 1], 2.0f0 * sino_def[6, 1]; atol=1e-5)

# Backprojection with VolumeGeometry
bp_def = backproject(sino_def, geom_par1; volume_size=(4, 4), backend=cpu())
bp_scaled = backproject(sino_def, geom_par1, vg_scaled; backend=cpu())
bp_scaled_kw = backproject(sino_def, geom_par1; volume_geometry=vg_scaled, backend=cpu())
bp_scaled_win = backproject(sino_def, geom_par1; volume_size=(4, 4), window=(-4, 4, -4, 4), backend=cpu())
@test size(bp_scaled) == (4, 4)
@test bp_scaled == bp_scaled_kw
@test bp_scaled == bp_scaled_win

# In-place backproject! with VolumeGeometry
bp_inplace = zeros(Float32, 4, 4)
backproject!(bp_inplace, sino_def, geom_par1, vg_scaled; backend=cpu())
@test bp_inplace == bp_scaled

# Reconstruction with VolumeGeometry (FBP & SIRT on CPU)
geom_par_multi = Parallel2DGeometry(Float32[0.0, pi/4, pi/2, 3pi/4]; detector_count=8, detector_width=1.0f0)
vol_test = reshape(Float32.(1:16), 4, 4)
sino_test = project(vol_test, geom_par_multi, vg_scaled; backend=cpu())

rec_fbp_vg = reconstruct(sino_test, geom_par_multi, vg_scaled, FBPMethod(); backend=cpu())
rec_fbp_kw = reconstruct(sino_test, geom_par_multi, FBPMethod(); volume_geometry=vg_scaled, backend=cpu())
@test size(rec_fbp_vg) == (4, 4)
@test rec_fbp_vg == rec_fbp_kw

rec_sirt_vg = reconstruct(sino_test, geom_par_multi, vg_scaled, SIRTMethod(iterations=5); backend=cpu())
rec_sirt_kw = reconstruct(sino_test, geom_par_multi, SIRTMethod(iterations=5); volume_geometry=vg_scaled, backend=cpu())
@test size(rec_sirt_vg) == (4, 4)
@test rec_sirt_vg == rec_sirt_kw

# In-place reconstruct! with VolumeGeometry
rec_inplace = zeros(Float32, 4, 4)
reconstruct!(rec_inplace, sino_test, geom_par_multi, vg_scaled, SIRTMethod(iterations=5); backend=cpu())
@test isapprox(rec_inplace, rec_sirt_vg; atol=1e-6)

# AstraProjectorMap with VolumeGeometry
map_vg = AstraProjectorMap(geom_par1, vg_scaled; backend=cpu())
@test map_vg.volume_size == (4, 4)
@test map_vg.volume_geometry == vg_scaled
sino_from_map = map_vg * vol4x4
@test sino_from_map == sino_scaled
bp_from_map = map_vg' * sino_def
@test bp_from_map == bp_scaled

# -------------------------------------------------------------------------
# Test DART (Discrete Algebraic Reconstruction Technique)
# -------------------------------------------------------------------------

# 1. dart_segment and dart_segment!
test_vol = Float32[0.1 0.4 0.9; 1.2 1.8 2.1]
seg_out = dart_segment(test_vol, [0.0, 1.0, 2.0])
@test seg_out == Float32[0.0 0.0 1.0; 1.0 2.0 2.0]

# Custom thresholds
seg_thresh = dart_segment(test_vol, [0.0, 1.0, 2.0]; thresholds=[0.3, 1.5])
@test seg_thresh == Float32[0.0 1.0 1.0; 1.0 2.0 2.0]

# In-place segmentation
seg_inplace = zeros(Float32, size(test_vol))
dart_segment!(seg_inplace, test_vol, [0.0, 1.0, 2.0])
@test seg_inplace == seg_out

# 2. dart_mask and dart_mask! (2D)
seg2d = zeros(Float32, 16, 16)
seg2d[6:11, 6:11] .= 1.0f0 # Central square

mask2d_cpu = dart_mask(seg2d; connectivity=8, radius=1, backend=cpu())
@test size(mask2d_cpu) == (16, 16)
# Center of square (e.g. (8,8)) should be fixed (0.0)
@test mask2d_cpu[8, 8] == 0.0f0
# Far outside (e.g. (2,2)) should be fixed (0.0)
@test mask2d_cpu[2, 2] == 0.0f0
# Boundary of square (e.g. (6,6), (6,8), (11,11)) should be free (1.0)
@test mask2d_cpu[6, 6] == 1.0f0
@test mask2d_cpu[6, 8] == 1.0f0
@test mask2d_cpu[11, 11] == 1.0f0

# Compare CPU and CUDA 2D mask
mask2d_conn4_cpu = dart_mask(seg2d; connectivity=4, radius=1, backend=cpu())
mask2d_r2_cpu = dart_mask(seg2d; connectivity=8, radius=2, threshold=2, backend=cpu())

if cuda_available
    mask2d_cuda = dart_mask(seg2d; connectivity=8, radius=1, backend=cuda())
    @test mask2d_cuda == mask2d_cpu

    mask2d_conn4_cuda = dart_mask(seg2d; connectivity=4, radius=1, backend=cuda())
    @test mask2d_conn4_cuda == mask2d_conn4_cpu

    mask2d_r2_cuda = dart_mask(seg2d; connectivity=8, radius=2, threshold=2, backend=cuda())
    @test mask2d_r2_cuda == mask2d_r2_cpu
end

# Random free voxels
rng_fixed = Random.MersenneTwister(42)
mask2d_rand = dart_mask(seg2d; connectivity=8, radius=1, p_free=0.1, backend=cpu(), rng=rng_fixed)
@test sum(mask2d_rand) > sum(mask2d_cpu)

# 3. dart_mask (3D)
seg3d = zeros(Float32, 12, 12, 12)
seg3d[5:8, 5:8, 5:8] .= 1.0f0

mask3d_cpu = dart_mask(seg3d; connectivity=26, radius=1, backend=cpu())
@test size(mask3d_cpu) == (12, 12, 12)
@test mask3d_cpu[6, 6, 6] == 0.0f0
@test mask3d_cpu[2, 2, 2] == 0.0f0
@test mask3d_cpu[5, 5, 5] == 1.0f0
@test mask3d_cpu[8, 8, 8] == 1.0f0

if cuda_available
    mask3d_cuda = dart_mask(seg3d; connectivity=26, radius=1, backend=cuda())
    @test mask3d_cuda == mask3d_cpu
end

# 4. dart_smoothing (2D & 3D)
img2d = zeros(Float32, 16, 16)
img2d[8, 8] = 10.0f0
smooth2d_cpu = dart_smoothing(img2d; b=0.2, radius=1, backend=cpu())
@test smooth2d_cpu[8, 8] < 10.0f0 # Center reduced
@test smooth2d_cpu[8, 9] > 0.0f0  # Neighbors received energy

vol3d = zeros(Float32, 10, 10, 10)
vol3d[5, 5, 5] = 10.0f0
smooth3d_cpu = dart_smoothing(vol3d; b=0.2, radius=1, backend=cpu())
@test smooth3d_cpu[5, 5, 5] < 10.0f0
@test smooth3d_cpu[5, 5, 6] > 0.0f0

if cuda_available
    smooth2d_cuda = dart_smoothing(img2d; b=0.2, radius=1, backend=cuda())
    @test isapprox(smooth2d_cpu, smooth2d_cuda; atol=1e-5)

    smooth3d_cuda = dart_smoothing(vol3d; b=0.2, radius=1, backend=cuda())
    @test isapprox(smooth3d_cpu, smooth3d_cuda; atol=1e-5)
end

# 5. Full 2D DART Reconstruction (CPU & CUDA)
# Create a discrete binary phantom
phantom2d = zeros(Float32, 32, 32)
phantom2d[10:23, 10:23] .= 1.0f0

dart_geom2d = Parallel2DGeometry(range(0, pi, length=24); detector_count=32, detector_width=1.0f0)
vol_geom2d = VolumeGeometry2D(32, 32)
dart_sino2d = project(phantom2d, dart_geom2d; volume_geometry=vol_geom2d, backend=cpu())

# Reconstruct using DART on CPU
dart_res_cpu = reconstruct_dart(
    dart_sino2d,
    dart_geom2d,
    [0.0f0, 1.0f0];
    volume_geometry=vol_geom2d,
    iterations=5,
    initial_iterations=20,
    arm_iterations=5,
    backend=cpu(),
)
@test dart_res_cpu.reconstruction isa Array{Float32, 2}
@test dart_res_cpu.segmentation isa Array{Float32, 2}
@test dart_res_cpu.mask isa Array{Float32, 2}
# Check that segmented phantom accurately matches ground truth
overlap_cpu = sum(dart_res_cpu.segmentation .== phantom2d) / length(phantom2d)
@test overlap_cpu > 0.95

if cuda_available
    # Reconstruct using DART on CUDA
    dart_res_cuda = reconstruct_dart(
        dart_sino2d,
        dart_geom2d,
        [0.0f0, 1.0f0];
        volume_geometry=vol_geom2d,
        iterations=5,
        initial_iterations=20,
        arm_iterations=5,
        backend=cuda(),
    )
    overlap_cuda = sum(dart_res_cuda.segmentation .== phantom2d) / length(phantom2d)
    @test overlap_cuda > 0.95

    # 6. Full 3D DART Reconstruction (CUDA)
    phantom3d = zeros(Float32, 16, 16, 16)
    phantom3d[5:12, 5:12, 5:12] .= 1.0f0

    dart_geom3d = Parallel3DGeometry(
        range(0, pi, length=16);
        detector_count_u=16,
        detector_count_v=16,
    )
    vol_geom3d = VolumeGeometry3D(16, 16, 16)
    dart_sino3d = project(phantom3d, dart_geom3d; volume_geometry=vol_geom3d, backend=cuda())

    dart_res_3d = reconstruct_dart(
        dart_sino3d,
        dart_geom3d,
        [0.0f0, 1.0f0];
        volume_geometry=vol_geom3d,
        iterations=3,
        initial_iterations=15,
        arm_iterations=3,
        backend=cuda(),
    )
    @test dart_res_3d.reconstruction isa Array{Float32, 3}
    @test dart_res_3d.segmentation isa Array{Float32, 3}
    @test dart_res_3d.mask isa Array{Float32, 3}
    overlap_3d = sum(dart_res_3d.segmentation .== phantom3d) / length(phantom3d)
    @test overlap_3d > 0.95
else
    @warn "CUDA not available; skipping DART CUDA reconstruction tests"
end

# =========================================================================
# 3D Composite FP & BP (Tiling & Multi-part Projection)
# =========================================================================
@testset "3D Composite Projection & Backprojection" begin
    # 1. Type validation and parts creation
    v_test = zeros(Float32, 16, 16, 16)
    vg_test = VolumeGeometry3D(16, 16, 16)
    vp = CompositeVolumePart(v_test, vg_test)
    @test vp.data === v_test
    @test vp.geometry === vg_test
    @test_throws DimensionMismatch CompositeVolumePart(zeros(Float32, 10, 10, 10), vg_test)

    angles_test = Float32.(range(0, pi, length=8))
    pg_test = Parallel3DGeometry(angles_test; detector_count_u=20, detector_count_v=16)
    sino_test = zeros(Float32, 20, 8, 16)
    pp = CompositeProjectionPart(sino_test, pg_test)
    @test pp.data === sino_test
    @test pp.geometry === pg_test
    @test_throws DimensionMismatch CompositeProjectionPart(zeros(Float32, 10, 8, 16), pg_test)

    # Aliases
    @test composite_project === do_composite_FP
    @test composite_project! === do_composite_FP!
    @test composite_backproject === do_composite_BP
    @test composite_backproject! === do_composite_BP!

    if cuda_available
        # 2. Multi-volume multi-resolution composite FP
        v1 = ones(Float32, 16, 16, 16)
        vg1 = VolumeGeometry3D(16, 16, 16; window=(-32.0f0, 0.0f0, -16.0f0, 16.0f0, -16.0f0, 16.0f0))
        v2 = ones(Float32, 16, 16, 16)
        vg2 = VolumeGeometry3D(16, 16, 16; window=(0.0f0, 32.0f0, -16.0f0, 16.0f0, -16.0f0, 16.0f0))

        angles1 = Float32.(range(0, pi/2, length=8))
        angles2 = Float32.(range(pi/2, pi, length=8))
        pg1 = Parallel3DGeometry(angles1; detector_count_u=32, detector_count_v=16)
        pg2 = Parallel3DGeometry(angles2; detector_count_u=32, detector_count_v=16)

        p1 = zeros(Float32, 32, 8, 16)
        p2 = zeros(Float32, 32, 8, 16)

        # In-place composite FP into 2 projection sets
        do_composite_FP!([p1 => pg1, p2 => pg2], [v1 => vg1, v2 => vg2])
        @test sum(p1) > 0.0f0
        @test sum(p2) > 0.0f0
        # Both halves have symmetric geometry & data, so sums should match closely
        @test isapprox(sum(p1), sum(p2); rtol=1e-3)

        # 3. Multi-volume composite BP
        v_bp1 = zeros(Float32, 16, 16, 16)
        v_bp2 = zeros(Float32, 16, 16, 16)
        do_composite_BP!([v_bp1 => vg1, v_bp2 => vg2], [p1 => pg1, p2 => pg2])
        @test sum(v_bp1) > 0.0f0
        @test sum(v_bp2) > 0.0f0
        @test isapprox(sum(v_bp1), sum(v_bp2); rtol=1e-3)

        # 4. Out-of-place composite FP and BP
        p_out = do_composite_FP([v1 => vg1, v2 => vg2], pg1)
        @test size(p_out) == (32, 8, 16)
        @test isapprox(p_out, p1)

        v_out = do_composite_BP(vg1, p1 => pg1)
        @test size(v_out) == (16, 16, 16)

        # 5. Accumulation mode (mode=:add, accumulate_FP!, accumulate_BP!)
        p_accum = copy(p1)
        accumulate_FP!(p_accum => pg1, [v1 => vg1, v2 => vg2])
        @test isapprox(p_accum, 2.0f0 .* p1)

        v_accum = copy(v_bp1)
        accumulate_BP!(v_accum => vg1, [p1 => pg1, p2 => pg2])
        @test isapprox(v_accum, 2.0f0 .* v_bp1)

        # 6. Bitwise consistency with standard direct project / backproject
        v_single = rand(Float32, 16, 16, 16)
        vg_single = VolumeGeometry3D(16, 16, 16)
        pg_single = Parallel3DGeometry(Float32.(range(0, pi, length=12)); detector_count_u=20, detector_count_v=16)

        p_std = project(v_single, pg_single; backend=cuda())
        p_comp = do_composite_FP(v_single => vg_single, pg_single)
        @test p_std == p_comp

        v_bp_std = backproject(p_std, pg_single; volume_geometry=vg_single, backend=cuda())
        v_bp_comp = do_composite_BP(vg_single, p_std => pg_single)
        @test isapprox(v_bp_std, v_bp_comp; atol=1e-4)

        # Cone3D consistency
        cg_single = Cone3DGeometry(Float32.(range(0, 2pi, length=10)); detector_count_u=24, detector_count_v=24, origin_source_distance=100.0f0, origin_detector_distance=50.0f0)
        p_cone_std = project(v_single, cg_single; backend=cuda())
        p_cone_comp = do_composite_FP(v_single => vg_single, cg_single)
        @test p_cone_std == p_cone_comp

        # ParallelVec3D consistency
        pvg_single = geom_2vec(pg_single)
        p_vec_std = project(v_single, pvg_single; backend=cuda())
        p_vec_comp = do_composite_FP(v_single => vg_single, pvg_single)
        @test p_vec_std == p_vec_comp

        # 8. Error handling
        @test_throws ArgumentError do_composite_FP!(p1 => pg1, v1 => vg1; mode=:invalid)
        @test_throws ArgumentError do_composite_BP!(v1 => vg1, p1 => pg1; mode=:invalid)
    else
        @warn "CUDA not available; skipping 3D composite execution tests"
    end
end

@testset "Geometry utilities: move_vol_geom, geom_size, project_point, projected_bbox" begin
    # 1. move_vol_geom 2D
    vg2 = VolumeGeometry2D(128, 128)
    @test center(vg2) == (0.0f0, 0.0f0)
    @test pixel_size(vg2) == (1.0f0, 1.0f0)

    vg2_shifted = move_vol_geom(vg2, (10.0, -5.0))
    @test center(vg2_shifted) == (10.0f0, -5.0f0)
    @test pixel_size(vg2_shifted) == (1.0f0, 1.0f0)
    @test vg2_shifted.window_min_x == -54.0f0
    @test vg2_shifted.window_max_x == 74.0f0
    @test vg2_shifted.window_min_y == -69.0f0
    @test vg2_shifted.window_max_y == 59.0f0

    # Varargs
    vg2_v = move_vol_geom(vg2, 10.0, -5.0)
    @test vg2_v == vg2_shifted

    # Relative shift
    vg2_rel = move_vol_geom(vg2_shifted, (2.0, 3.0); is_relative=true)
    @test center(vg2_rel) == (12.0f0, -2.0f0)

    # 2. move_vol_geom 3D
    vg3 = VolumeGeometry3D(64, 64, 32)
    @test center(vg3) == (0.0f0, 0.0f0, 0.0f0)
    @test voxel_size(vg3) == (1.0f0, 1.0f0, 1.0f0)

    vg3_shifted = move_vol_geom(vg3, (1.0, 2.0, 3.0))
    @test center(vg3_shifted) == (1.0f0, 2.0f0, 3.0f0)
    @test voxel_size(vg3_shifted) == (1.0f0, 1.0f0, 1.0f0)
    @test vg3_shifted.window_min_z == -13.0f0
    @test vg3_shifted.window_max_z == 19.0f0

    vg3_rel = move_vol_geom(vg3, (1.0, 1.0, 1.0); is_relative=true)
    @test center(vg3_rel) == (1.0f0, 1.0f0, 1.0f0)

    # 3. size / geom_size / sinogram_size
    pg2 = Parallel2DGeometry(Float32[0.0, 1.0], 100, 1.0f0)
    fg2 = FanFlat2DGeometry(Float32[0.0, 1.0], 80, 1.0f0, 500.0f0, 250.0f0)
    pvg2 = geom_2vec(pg2)
    fvg2 = geom_2vec(fg2)

    @test size(pg2) == (100, 2)
    @test geom_size(pg2) == (100, 2)
    @test sinogram_size(pg2) == (100, 2)
    @test geom_size(pg2, 1) == 100
    @test geom_size(pg2, 2) == 2

    @test size(fg2) == (80, 2)
    @test geom_size(fg2) == (80, 2)
    @test size(pvg2) == (100, 2)
    @test size(fvg2) == (80, 2)

    @test geom_size(vg2) == (128, 128)
    @test geom_size(vg3) == (64, 64, 32)
    @test geom_size(vg3, 3) == 32

    pg3 = Parallel3DGeometry(Float32[0.0, 1.0], 100, 50, 1.0f0, 1.0f0)
    cg3 = Cone3DGeometry(Float32[0.0, 1.0], 120, 60, 1.0f0, 1.0f0, 1000.0f0, 500.0f0)
    pvg3 = geom_2vec(pg3)
    cvg3 = geom_2vec(cg3)

    @test size(pg3) == (100, 2, 50)
    @test geom_size(pg3) == (100, 2, 50)
    @test sinogram_size(pg3) == (100, 2, 50)
    @test size(cg3) == (120, 2, 60)
    @test size(pvg3) == (100, 2, 50)
    @test size(cvg3) == (120, 2, 60)

    # 4. project_point (Parallel3D)
    # Origin at center of detector: (det_u - 1) / 2 = 49.5, (det_v - 1) / 2 = 24.5 in 0-based
    u0, v0 = project_point(pg3, 0.0, 0.0, 0.0, 0; zero_based=true)
    @test isapprox(u0, 49.5)
    @test isapprox(v0, 24.5)

    u1, v1 = project_point(pg3, 0.0, 0.0, 0.0, 1; zero_based=false)
    @test isapprox(u1, 50.5)
    @test isapprox(v1, 25.5)

    # 5. project_point (Cone3D)
    cu0, cv0 = project_point(cg3, 0.0, 0.0, 0.0, 0; zero_based=true)
    @test isapprox(cu0, 59.5)
    @test isapprox(cv0, 29.5)

    # 6. project_point (ParallelVec3D & ConeVec3D)
    pvu0, pvv0 = project_point(pvg3, 0.0, 0.0, 0.0, 0; zero_based=true)
    @test isapprox(pvu0, 49.5)
    @test isapprox(pvv0, 24.5)

    cvu0, cvv0 = project_point(cvg3, 0.0, 0.0, 0.0, 0; zero_based=true)
    @test isapprox(cvu0, 59.5)
    @test isapprox(cvv0, 29.5)

    # 7. projected_bbox
    bbox = projected_bbox(pg3, -10.0, 10.0, -10.0, 10.0, -5.0, 5.0; zero_based=true)
    @test bbox.min_u < bbox.max_u
    @test bbox.min_v < bbox.max_v
    # Bounding box must contain origin projection
    @test bbox.min_u <= 49.5 <= bbox.max_u
    @test bbox.min_v <= 24.5 <= bbox.max_v

    # 1-based bbox
    bbox_1 = projected_bbox(pg3, -10.0, 10.0, -10.0, 10.0, -5.0, 5.0; zero_based=false)
    @test isapprox(bbox_1.min_u, bbox.min_u + 1.0)
    @test isapprox(bbox_1.max_u, bbox.max_u + 1.0)
    @test isapprox(bbox_1.min_v, bbox.min_v + 1.0)
    @test isapprox(bbox_1.max_v, bbox.max_v + 1.0)

    # Single angle bbox
    bbox_single = projected_bbox(pg3, -10.0, 10.0, -10.0, 10.0, -5.0, 5.0; angle=1, zero_based=false)
    @test bbox_single.min_u <= bbox_single.max_u
    @test bbox_single.min_v <= bbox_single.max_v

    # VolumeGeometry3D overload
    bbox_vg = projected_bbox(pg3, vg3; zero_based=false)
    @test bbox_vg.min_u < bbox_vg.max_u
    @test bbox_vg.min_v < bbox_vg.max_v

    # Tuple overload
    bbox_t = projected_bbox(pg3, (-10.0, 10.0, -10.0, 10.0, -5.0, 5.0); zero_based=false)
    @test bbox_t == bbox_1
end




