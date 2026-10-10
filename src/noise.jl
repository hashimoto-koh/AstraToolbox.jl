"""
    add_noise_to_sino!(sinogram::Array{Float32}, I0::Real; seed=nothing, rng=nothing)

Adds Poisson noise to a sinogram in-place.

# Arguments
- `sinogram`: 2D or 3D array of the sinogram.
- `I0`: Photon intensity (the higher the intensity, the lower the relative noise).
- `seed`: Optional seed for the random number generator.
- `rng`: Optional `AbstractRNG` object.

# Returns
- `sinogram`: The updated sinogram array with Poisson noise applied (mutated in-place).

# Examples
```julia
add_noise_to_sino!(sino, 1e5; seed=42)
```

The logic follows the upstream Python version:
1. `max_val = max(sinogram)`
2. `sinogram_scaled = sinogram / max_val`
3. `sinogram_ct = I0 * exp(-sinogram_scaled)`
4. `sinogram_ct_poisson = rand(Poisson(sinogram_ct))`
5. `sinogram_ct_d = sinogram_ct_poisson / I0`
6. `sinogram_out = -max_val * log(sinogram_ct_d)`
"""
function add_noise_to_sino!(sinogram::Array{Float32}, I0::Real; seed=nothing, rng=nothing)
    if rng === nothing
        if seed !== nothing
            rng = Random.MersenneTwister(seed)
        else
            rng = Random.default_rng()
        end
    end

    max_val = maximum(sinogram)
    @inbounds for i in eachindex(sinogram)
        val = sinogram[i]
        scaled = val / max_val
        ct = I0 * exp(-scaled)
        # Sample from Poisson distribution
        ct_poisson = rand(rng, Poisson(ct))
        # Convert back to density
        ct_d = ct_poisson / I0
        # If Poisson sample is 0, log will be -Inf. 
        # Python version uses np.log which handles 0 by returning -inf.
        sinogram[i] = -max_val * log(ct_d)
    end
    return sinogram
end

"""
    add_noise_to_sino(sinogram::AbstractArray, I0::Real; seed=nothing, rng=nothing)

Adds Poisson noise to a sinogram.

# Arguments
- `sinogram`: 2D or 3D array of the sinogram.
- `I0`: Photon intensity.
- `seed`: Optional seed.
- `rng`: Optional `AbstractRNG`.

# Returns
- `Array{Float32}`: A new noisy sinogram array.

# Examples
```julia
noisy_sino = add_noise_to_sino(sino, 1e5)
```
"""
function add_noise_to_sino(sinogram::AbstractArray, I0::Real; seed=nothing, rng=nothing)
    out = Array{Float32}(sinogram)
    return add_noise_to_sino!(out, I0; seed=seed, rng=rng)
end
