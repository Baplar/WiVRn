/*
 * WiVRn VR streaming
 * Copyright (C) 2022  Guillaume Meunier <guillaume.meunier@centraliens.net>
 * Copyright (C) 2022  Patrick Nicolas <patricknicolas@laposte.net>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#version 450

#ifdef VERT_SHADER

layout (constant_id = 0) const bool use_foveation_x = false;
layout (constant_id = 1) const bool use_foveation_y = false;
layout (constant_id = 2) const int nb_x = 64;
layout (constant_id = 3) const int nb_y = 64;

layout(set = 0, binding = 1) uniform UniformBufferObject
{
	vec2 a;
	vec2 b;
	vec2 lambda;
	vec2 xc;
}
ubo;

layout(location = 0) out vec2 outUV;

vec2 positions[6] = vec2[](
	vec2(0, 0), vec2(1, 0), vec2(0, 1),
	vec2(1, 0), vec2(0, 1), vec2(1, 1));

vec2 unfoveate(vec2 uv)
{
	uv = 2 * uv - 1;
	if (use_foveation_x && use_foveation_y)
	{
		uv = ubo.lambda * tan(ubo.a * uv + ubo.b) + ubo.xc;
	}
	else
	{
		if (use_foveation_x)
		{
			uv.x = (ubo.lambda * tan(ubo.a * uv + ubo.b) + ubo.xc).y;
		}
		if (use_foveation_y)
		{
			uv.y = (ubo.lambda * tan(ubo.a * uv + ubo.b) + ubo.xc).y;
		}
	}
	return uv;
}

void main()
{
	vec2 quad_size = 1 / vec2(nb_x, nb_y);
	int cell_id = gl_VertexIndex / 6;

	vec2 top_left = quad_size * vec2(cell_id % nb_x, cell_id / nb_x);
	outUV = top_left + positions[gl_VertexIndex % 6] * quad_size;

	gl_Position = vec4(unfoveate(outUV), 0.0, 1.0);
}
#endif

#ifdef FRAG_SHADER

precision mediump float;
precision highp int;

#define OperationMode 1

layout (constant_id = 0) const bool use_sgsr = false;
layout (constant_id = 1) const bool use_edge_direction = false;
layout (constant_id = 2) const float edge_threshold = 4.0/255.0;
layout (constant_id = 3) const float edge_sharpness = 2.0;

layout(set = 0, binding = 0) uniform mediump sampler2D texSampler;

layout(location = 0) in highp vec2 inUV;

layout(location = 0) out vec4 outColor;

float fastLanczos2(float x)
{
	float wA = x-4.0;
	float wB = x*wA-wA;
	wA *= wA;
	return wB*wA;
}

vec2 weightY(float dx, float dy, float c, vec3 data)
{
	float std = data.x;
	vec2 dir = data.yz;

	float edgeDis = ((dx*dir.y)+(dy*dir.x));
	float x = (((dx*dx)+(dy*dy))+((edgeDis*edgeDis)*((clamp(((c*c)*std),0.0,1.0)*0.7)+-1.0)));

	float w = fastLanczos2(x);
	return vec2(w, w * c);	
}

vec2 weightYned(float dx, float dy, float c, float std)
{
	float x = ((dx*dx)+(dy* dy))* 0.55 + clamp(abs(c)*std, 0.0, 1.0);

	float w = fastLanczos2(x);
	return vec2(w, w * c);	
}

vec2 edgeDirection(vec4 left, vec4 right)
{
	vec2 dir;
	float RxLz = (right.x + (-left.z));
	float RwLy = (right.w + (-left.y));
	vec2 delta;
	delta.x = (RxLz + RwLy);
	delta.y = (RxLz + (-RwLy));
	float lengthInv = inversesqrt((delta.x * delta.x+ 3.075740e-05) + (delta.y * delta.y));
	dir.x = (delta.x * lengthInv);
	dir.y = (delta.y * lengthInv);
	return dir;
}

vec4 textureSgsr(sampler2D texSampler, vec2 inUV)
{
	vec4 color = textureLod(texSampler,inUV,0.0).xyzw;
	float alpha = color.a;

	if (OperationMode == 1)
		color.a = 0.0;

	if (OperationMode != 4)
	{
		vec2 dim = textureSize(texSampler, 0);
		vec4 viewportInfo = vec4(1/dim.x, 1/dim.y, dim.x, dim.y);

		highp vec2 imgCoord = ((inUV.xy*viewportInfo.zw)+vec2(-0.5,0.5));
		highp vec2 imgCoordPixel = floor(imgCoord);
		highp vec2 coord = (imgCoordPixel*viewportInfo.xy);
		vec2 pl = (imgCoord+(-imgCoordPixel));
		vec4 left = textureGather(texSampler,coord, OperationMode);

		float edgeVote = abs(left.z - left.y) + abs(color[OperationMode] - left.y)  + abs(color[OperationMode] - left.z) ;
		if(edgeVote > edge_threshold)
		{
			coord.x += viewportInfo.x;

			vec4 right = textureGather(texSampler,coord + vec2(viewportInfo.x, 0.0), OperationMode);
			vec4 upDown;
			upDown.xy = textureGather(texSampler,coord + vec2(0.0, -viewportInfo.y),OperationMode).wz;
			upDown.zw  = textureGather(texSampler,coord + vec2(0.0, viewportInfo.y), OperationMode).yx;

			float mean = (left.y+left.z+right.x+right.w)*0.25;
			left = left - vec4(mean);
			right = right - vec4(mean);
			upDown = upDown - vec4(mean);
			color.w =color[OperationMode] - mean;

			float sum = (((((abs(left.x)+abs(left.y))+abs(left.z))+abs(left.w))+(((abs(right.x)+abs(right.y))+abs(right.z))+abs(right.w)))+(((abs(upDown.x)+abs(upDown.y))+abs(upDown.z))+abs(upDown.w)));				
			float sumMean = 1.014185e+01/sum;
			float std = (sumMean*sumMean);	

			vec2 aWY;
			if (use_edge_direction) {
				vec3 data = vec3(std, edgeDirection(left, right));
				aWY = weightY(pl.x, pl.y+1.0, upDown.x,data);				
				aWY += weightY(pl.x-1.0, pl.y+1.0, upDown.y,data);
				aWY += weightY(pl.x-1.0, pl.y-2.0, upDown.z,data);
				aWY += weightY(pl.x, pl.y-2.0, upDown.w,data);			
				aWY += weightY(pl.x+1.0, pl.y-1.0, left.x,data);
				aWY += weightY(pl.x, pl.y-1.0, left.y,data);
				aWY += weightY(pl.x, pl.y, left.z,data);
				aWY += weightY(pl.x+1.0, pl.y, left.w,data);
				aWY += weightY(pl.x-1.0, pl.y-1.0, right.x,data);
				aWY += weightY(pl.x-2.0, pl.y-1.0, right.y,data);
				aWY += weightY(pl.x-2.0, pl.y, right.z,data);
				aWY += weightY(pl.x-1.0, pl.y, right.w,data);
			} else {
				aWY = weightYned(pl.x, pl.y+1.0, upDown.x,std);				
				aWY += weightYned(pl.x-1.0, pl.y+1.0, upDown.y,std);
				aWY += weightYned(pl.x-1.0, pl.y-2.0, upDown.z,std);
				aWY += weightYned(pl.x, pl.y-2.0, upDown.w,std);			
				aWY += weightYned(pl.x+1.0, pl.y-1.0, left.x,std);
				aWY += weightYned(pl.x, pl.y-1.0, left.y,std);
				aWY += weightYned(pl.x, pl.y, left.z,std);
				aWY += weightYned(pl.x+1.0, pl.y, left.w,std);
				aWY += weightYned(pl.x-1.0, pl.y-1.0, right.x,std);
				aWY += weightYned(pl.x-2.0, pl.y-1.0, right.y,std);
				aWY += weightYned(pl.x-2.0, pl.y, right.z,std);
				aWY += weightYned(pl.x-1.0, pl.y, right.w,std);
			}

			float finalY = aWY.y/aWY.x;
			float maxY = max(max(left.y,left.z),max(right.x,right.w));
			float minY = min(min(left.y,left.z),min(right.x,right.w));
			float deltaY = clamp(edge_sharpness*finalY, minY, maxY) -color.w;			
			
			//smooth high contrast input
			deltaY = clamp(deltaY, -23.0 / 255.0, 23.0 / 255.0);

			color.x = clamp((color.x+deltaY),0.0,1.0);
			color.y = clamp((color.y+deltaY),0.0,1.0);
			color.z = clamp((color.z+deltaY),0.0,1.0);
		}
	}

	color.a = alpha;

	return color;
}

void main()
{
	if (use_sgsr)
	{
		outColor = textureSgsr(texSampler, inUV);
	}
	else
	{
		outColor = texture(texSampler, inUV);
	}
}
#endif

