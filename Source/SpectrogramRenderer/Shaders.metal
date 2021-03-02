#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float4 position [[position]];
    float2 uv;
};

vertex Vertex vertex_shader(constant float4 *vertices [[buffer(0)]],
                            uint id [[vertex_id]])
{
    return {
        .position = vertices[id],
        .uv = (vertices[id].xy + float2(1)) / float2(2)
    };
}

struct FragmentShaderParameters {
    float zoom;
    float heightOffset;
    float heightOverflow;
};

fragment float4 fragment_shader(Vertex vtx [[stage_in]],
                                constant FragmentShaderParameters& params [[buffer(0)]],
                                texture2d<float> field [[texture(0)]])
{
    constexpr sampler smplr(coord::normalized,
                            address::clamp_to_zero,
                            filter::linear);

    // Pinching to zoom towards the newest part of the spectrogram
    float2 zommed = float2(1.0, params.zoom) * vtx.uv;

    // Reverse the spectrogram on the screen (so new information is comming in from the right)
    float2 reversed = float2(zommed.r, 1.0 - zommed.g);

    // Make sure the smooth buffer fill is not on screen
    float2 overflowed = float2(1.0, 1.0 - params.heightOverflow) * reversed;

    // Offset for the ring buffer
    float2 offset = float2(overflowed.r, fmod(overflowed.g + params.heightOffset, 1.0));

    float cell = field.sample(smplr, offset).r;
    return float4(cell);
}


