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

fragment float4 fragment_shader(Vertex vtx [[stage_in]],
                                texture2d<float> field [[texture(0)]])
{
    constexpr sampler smplr(coord::normalized,
                            address::clamp_to_zero,
                            filter::nearest);
    float cell = field.sample(smplr, vtx.uv).r;
    return float4(cell);
}


