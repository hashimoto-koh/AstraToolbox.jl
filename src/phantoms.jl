# Shepp-Logan phantom generators (2D and 3D)
# Implements the analytical ellipsoid rasterization following ASTRA Toolbox upstream (SheppLogan.cpp)

struct Ellipse2D
    x::Float64
    y::Float64
    axisx::Float64
    axisy::Float64
    rotation::Float64
    value::Float64
end

struct Ellipsoid3D
    x::Float64
    y::Float64
    z::Float64
    axisx::Float64
    axisy::Float64
    axisz::Float64
    rotation::Float64
    value::Float64
end

function _default_ellipses_2d(modified::Bool)
    values = modified ?
        Float64[1.0, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] :
        Float64[2.0, -0.98, -0.02, -0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

    return Ellipse2D[
        Ellipse2D( 0.0,    0.0,     0.69,   0.92,    0.0, values[1]),
        Ellipse2D( 0.0,   -0.0184,  0.6624, 0.874,   0.0, values[2]),
        Ellipse2D( 0.22,   0.0,     0.11,   0.31,  -18.0, values[3]),
        Ellipse2D(-0.22,   0.0,     0.16,   0.41,   18.0, values[4]),
        Ellipse2D( 0.0,    0.35,    0.21,   0.25,    0.0, values[5]),
        Ellipse2D( 0.0,    0.1,     0.046,  0.046,   0.0, values[6]),
        Ellipse2D( 0.0,   -0.1,     0.046,  0.046,   0.0, values[7]),
        Ellipse2D(-0.08,  -0.605,   0.046,  0.023,   0.0, values[8]),
        Ellipse2D( 0.0,   -0.605,   0.023,  0.023,   0.0, values[9]),
        Ellipse2D( 0.06,  -0.605,   0.023,  0.046,   0.0, values[10]),
    ]
end

function _default_ellipsoids_3d(modified::Bool)
    values = modified ?
        Float64[1.0, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] :
        Float64[2.0, -0.98, -0.02, -0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

    return Ellipsoid3D[
        Ellipsoid3D( 0.0,    0.0,     0.0, 0.69,   0.92,  0.81,   0.0, values[1]),
        Ellipsoid3D( 0.0,   -0.0184,  0.0, 0.6624, 0.874, 0.78,   0.0, values[2]),
        Ellipsoid3D( 0.22,   0.0,     0.0, 0.11,   0.31,  0.22, -18.0, values[3]),
        Ellipsoid3D(-0.22,   0.0,     0.0, 0.16,   0.41,  0.28,  18.0, values[4]),
        Ellipsoid3D( 0.0,    0.35,    0.0, 0.21,   0.25,  0.41,   0.0, values[5]),
        Ellipsoid3D( 0.0,    0.1,     0.0, 0.046,  0.046, 0.05,   0.0, values[6]),
        Ellipsoid3D( 0.0,   -0.1,     0.0, 0.046,  0.046, 0.05,   0.0, values[7]),
        Ellipsoid3D(-0.08,  -0.605,   0.0, 0.046,  0.023, 0.05,   0.0, values[8]),
        Ellipsoid3D( 0.0,   -0.605,   0.0, 0.023,  0.023, 0.02,   0.0, values[9]),
        Ellipsoid3D( 0.06,  -0.605,   0.0, 0.023,  0.046, 0.02,   0.0, values[10]),
    ]
end

function _add_ellipse!(data::Array{Float32, 2}, width::Int, height::Int, ell::Ellipse2D)
    a = (0.5 + 0.5 * ell.x) * width - 0.5
    b = (0.5 - 0.5 * ell.y) * height - 0.5
    c = 0.5 * ell.axisx * width
    d = 0.5 * ell.axisy * height
    th = ell.rotation * π / 180.0

    costh = cos(th)
    costh2 = costh * costh
    sinth = sin(th)
    sinth2 = sinth * sinth
    a2 = a * a
    b2 = b * b
    c2 = c * c
    d2 = d * d

    c20 = costh2 / c2 + sinth2 / d2
    c10 = -2 * a * costh2 / c2 + 2 * b * costh * sinth / c2 - 2 * b * costh * sinth / d2 - 2 * a * sinth2 / d2
    c11 = -2 * costh * sinth / c2 + 2 * costh * sinth / d2
    c00 = a2 * costh2 / c2 + b2 * costh2 / d2 - 2 * a * b * costh * sinth / c2 + 2 * a * b * costh * sinth / d2 + b2 * sinth2 / c2 + a2 * sinth2 / d2 - 1.0
    c01 = -2 * b * costh2 / d2 + 2 * a * costh * sinth / c2 - 2 * a * costh * sinth / d2 - 2 * b * sinth2 / c2
    c02 = costh2 / d2 + sinth2 / c2

    val = Float32(ell.value)

    @inbounds for y in 0:(height - 1)
        A = c20
        B = y * c11 + c10
        C = y * y * c02 + y * c01 + c00
        D = B * B - 4.0 * A * C
        D < 0.0 && continue

        sqrtD = sqrt(D)
        xmin = Int(ceil(0.5 * (-B - sqrtD) / A))
        xmax = Int(floor(0.5 * (-B + sqrtD) / A)) + 1

        xmin = max(xmin, 0)
        xmax = min(xmax, width)

        (xmin >= width || xmax <= 0 || xmax <= xmin) && continue

        y_idx = y + 1
        for x in xmin:(xmax - 1)
            data[x + 1, y_idx] += val
        end
    end
end

function _add_ellipsoid!(data::Array{Float32, 3}, width::Int, height::Int, depth::Int, ell::Ellipsoid3D)
    a = (0.5 + 0.5 * ell.x) * width - 0.5
    b = (0.5 - 0.5 * ell.y) * height - 0.5
    c = 0.5 * ell.axisx * width
    d = 0.5 * ell.axisy * height
    e = (0.5 - 0.5 * ell.z) * depth - 0.5
    f = 0.5 * ell.axisz * depth
    th = ell.rotation * π / 180.0

    costh = cos(th)
    costh2 = costh * costh
    sinth = sin(th)
    sinth2 = sinth * sinth
    a2 = a * a
    b2 = b * b
    c2 = c * c
    d2 = d * d

    c20 = costh2 / c2 + sinth2 / d2
    c10 = -2 * a * costh2 / c2 + 2 * b * costh * sinth / c2 - 2 * b * costh * sinth / d2 - 2 * a * sinth2 / d2
    c11 = -2 * costh * sinth / c2 + 2 * costh * sinth / d2
    c00 = a2 * costh2 / c2 + b2 * costh2 / d2 - 2 * a * b * costh * sinth / c2 + 2 * a * b * costh * sinth / d2 + b2 * sinth2 / c2 + a2 * sinth2 / d2 - 1.0
    c01 = -2 * b * costh2 / d2 + 2 * a * costh * sinth / c2 - 2 * a * costh * sinth / d2 - 2 * b * sinth2 / c2
    c02 = costh2 / d2 + sinth2 / c2

    val = Float32(ell.value)

    @inbounds for y in 0:(height - 1)
        A = c20
        B = y * c11 + c10
        C = y * y * c02 + y * c01 + c00
        D = B * B - 4.0 * A * C
        D < 0.0 && continue

        sqrtD = sqrt(D)
        xmin = Int(ceil(0.5 * (-B - sqrtD) / A))
        xmax = Int(floor(0.5 * (-B + sqrtD) / A)) + 1

        xmin = max(xmin, 0)
        xmax = min(xmax, width)

        (xmin >= width || xmax <= 0 || xmax <= xmin) && continue

        y_idx = y + 1
        for x in xmin:(xmax - 1)
            x_val = Float64(x)
            xy = A * x_val * x_val + B * x_val + C
            xy >= 0.0 && continue

            sq_val = f * sqrt(-xy)
            zmin = Int(ceil(e - sq_val))
            zmax = Int(floor(e + sq_val)) + 1

            zmin = max(zmin, 0)
            zmax = min(zmax, depth)

            (zmin >= depth || zmax <= 0 || zmax <= zmin) && continue

            x_idx = x + 1
            for z in zmin:(zmax - 1)
                data[x_idx, y_idx, z + 1] += val
            end
        end
    end
end

"""
    shepp_logan!(out::Array{Float32, 2}; modified::Bool=true)

Rasterizes the 2D Shepp-Logan phantom directly into the provided `out` buffer in-place.
All negative values are clamped to `0f0` matching upstream ASTRA behavior.

# Arguments
- `out`: 2D `Array{Float32, 2}` with dimensions `(nx, ny)`.
- `modified`: If `true` (default), generates the Modified Shepp-Logan phantom with enhanced contrast (Kak & Slaney 1988). If `false`, generates the original Shepp & Logan (1974) phantom.

# Returns
- `out`: The modified array `out` (mutated in-place).

# Examples
```julia
out = zeros(Float32, 256, 256)
shepp_logan!(out)
```
"""
function shepp_logan!(out::Array{Float32, 2}; modified::Bool=true)
    fill!(out, 0.0f0)
    width, height = size(out)
    ells = _default_ellipses_2d(modified)
    for ell in ells
        _add_ellipse!(out, width, height, ell)
    end
    @inbounds for i in eachindex(out)
        if out[i] < 0.0f0
            out[i] = 0.0f0
        end
    end
    return out
end

"""
    shepp_logan(dims::Tuple{Integer, Integer}; modified::Bool=true)
    shepp_logan(size::Integer=256; modified::Bool=true)

Generates a 2D Shepp-Logan phantom of the specified dimensions.

# Arguments
- `dims`: Tuple `(nx, ny)` specifying width and height.
- `size`: Integer specifying `(size, size)`. Defaults to `256`.
- `modified`: If `true` (default), generates the Modified Shepp-Logan phantom.

# Returns
- `Array{Float32, 2}`: 2D array of size `(nx, ny)` containing phantom densities.

# Examples
```julia
phantom = shepp_logan((256, 256))
phantom_orig = shepp_logan(256; modified=false)
```
"""
function shepp_logan(dims::Tuple{Integer, Integer}; modified::Bool=true)
    out = zeros(Float32, Int(dims[1]), Int(dims[2]))
    return shepp_logan!(out; modified=modified)
end

function shepp_logan(n::Integer=256; modified::Bool=true)
    n > 0 || throw(ArgumentError("size must be positive, got $n"))
    return shepp_logan((Int(n), Int(n)); modified=modified)
end

function shepp_logan(geometry::AbstractGeometry2D; volume_size::Tuple{Integer, Integer}=(256, 256), modified::Bool=true)
    return shepp_logan(volume_size; modified=modified)
end

"""
    shepp_logan!(out::Array{Float32, 3}; modified::Bool=true)

Rasterizes the 3D Shepp-Logan phantom directly into the provided `out` buffer in-place.
All negative values are clamped to `0f0` matching upstream ASTRA behavior.

# Arguments
- `out`: 3D `Array{Float32, 3}` with dimensions `(nx, ny, nz)`.
- `modified`: If `true` (default), generates the Modified Shepp-Logan phantom.

# Returns
- `out`: The modified array `out` (mutated in-place).

# Examples
```julia
out = zeros(Float32, 128, 128, 128)
shepp_logan!(out)
```
"""
function shepp_logan!(out::Array{Float32, 3}; modified::Bool=true)
    fill!(out, 0.0f0)
    width, height, depth = size(out)
    ells = _default_ellipsoids_3d(modified)
    for ell in ells
        _add_ellipsoid!(out, width, height, depth, ell)
    end
    @inbounds for i in eachindex(out)
        if out[i] < 0.0f0
            out[i] = 0.0f0
        end
    end
    return out
end

"""
    shepp_logan_3d(dims::Tuple{Integer, Integer, Integer}; modified::Bool=true)
    shepp_logan_3d(size::Integer=128; modified::Bool=true)
    shepp_logan(dims::Tuple{Integer, Integer, Integer}; modified::Bool=true)

Generates a 3D Shepp-Logan phantom of the specified dimensions.

# Arguments
- `dims`: Tuple `(nx, ny, nz)` specifying width, height, and depth.
- `size`: Integer specifying `(size, size, size)`. Defaults to `128`.
- `modified`: If `true` (default), generates the Modified Shepp-Logan phantom.

# Returns
- `Array{Float32, 3}`: 3D array of size `(nx, ny, nz)` containing phantom densities.

# Examples
```julia
volume = shepp_logan_3d((128, 128, 128))
volume_cube = shepp_logan_3d(64)
```
"""
function shepp_logan_3d(dims::Tuple{Integer, Integer, Integer}; modified::Bool=true)
    out = zeros(Float32, Int(dims[1]), Int(dims[2]), Int(dims[3]))
    return shepp_logan!(out; modified=modified)
end

function shepp_logan_3d(n::Integer=128; modified::Bool=true)
    n > 0 || throw(ArgumentError("size must be positive, got $n"))
    return shepp_logan_3d((Int(n), Int(n), Int(n)); modified=modified)
end

function shepp_logan(dims::Tuple{Integer, Integer, Integer}; modified::Bool=true)
    return shepp_logan_3d(dims; modified=modified)
end

function shepp_logan(geometry::AbstractGeometry3D; volume_size::Tuple{Integer, Integer, Integer}=(128, 128, 128), modified::Bool=true)
    return shepp_logan_3d(volume_size; modified=modified)
end
