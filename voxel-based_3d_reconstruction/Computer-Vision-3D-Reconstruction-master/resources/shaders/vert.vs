#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec3 aTangent;
layout (location = 4) in vec3 aInstancePos;
layout (location = 5) in vec3 in_Color;


out VS_OUT
{
    vec3 fragPos;
    vec3 normal;
    vec2 texCoords;
    vec3 tanLightPos;
    vec3 tanFragPos;
    vec3 tanViewPos;
    vec4 fragPosLightSpace;
    vec3 out_Color;
} vs_out;

uniform mat4 viewProject;
uniform mat4 model;
uniform mat4 rotation;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform mat4 lightSpaceMatrix;

void main()
{
    vs_out.out_Color = in_Color;
    vs_out.texCoords = aTexCoords;
    vs_out.fragPos = vec3(model * rotation * vec4(aPos, 1.0)) + aInstancePos;
    vs_out.normal = transpose(inverse(mat3(model * rotation))) * aNormal + aInstancePos;
    gl_Position = viewProject * vec4(vs_out.fragPos, 1.0);

    vec3 N = normalize(vec3(vec4(aNormal, 0.0) * model * rotation));
    vec3 T = normalize(vec3(vec4(aTangent, 0.0) * model * rotation));
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);
    mat3 TBN = transpose(mat3(T, B, N));

    vs_out.tanViewPos = TBN * viewPos;
    vs_out.tanLightPos = TBN * lightPos;
    vs_out.tanFragPos = TBN * vs_out.fragPos;
    vs_out.fragPosLightSpace = lightSpaceMatrix * vec4(vs_out.fragPos, 1.0);
}
