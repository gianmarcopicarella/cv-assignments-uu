#version 330 core
#define WEIGHT_LEN 5

out vec4 fragColor;

in vec2 texCoords;

uniform sampler2D image;
uniform bool horizontal;
uniform float weight[WEIGHT_LEN] = float[](0.105, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162);

void main()
{
    fragColor = vec4(texture(image, texCoords).rgb, 1.0);
}