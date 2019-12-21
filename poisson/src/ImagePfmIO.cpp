/*
 *  Copyright (c) 2016, Marco Manzi and Markus Kettunen. All rights reserved.
 *  Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *     *  Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *     *  Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *     *  Neither the name of the NVIDIA CORPORATION nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ImagePfmIO.hpp"
#include <stdio.h>
#include <stdlib.h>

using namespace poisson;

//------------------------------------------------------------------------

const char* poisson::importPfmImage(Vec3f** pixelsOut, int* widthOut, int* heightOut, const char* path)
{
    assert(pixelsOut && widthOut && heightOut && path);
    *pixelsOut = NULL;
    *widthOut = 0;
    *heightOut = 0;

    // Open file.

    FILE* f = NULL;
    if (fopen_s(&f, path, "rb") != 0)
        return "Failed to open file!";

    // Read header fields.

    const int numFields = 4;
    std::string fields[numFields];

    for (int fieldIdx = 0; fieldIdx < numFields;)
    {
        // Read character.

        int c = fgetc(f);
        if (c <= 0)
        {
            fclose(f);
            return "Failed to read file!";
        }

        // Whitespace => advance to the next field.

        if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
        {
            fieldIdx++;
            continue;
        }

        // Field is too long => error.

        if (fields[fieldIdx].length() >= 1024)
        {
            fclose(f);
            return "Corrupt PFM header!";
        }

        // Append character to the field.

        fields[fieldIdx] += (char)c;
    }

    // Parse field 0: format identifier.

    int numChannels = -1;
    if (fields[0] == "PF") // RGB
        numChannels = 3;
    else if (fields[0] == "Pf") // grayscale
        numChannels = 1;
    else
    {
        fclose(f);
        return "Invalid PFM format identifier!";
    }

    // Parse fields 1--2: width & height.

    int width = std::stoi(fields[1]);
    int height = std::stoi(fields[2]);

    if (width < 1 || height < 1)
    {
        fclose(f);
        return "Invalid PFM dimensions!";
    }

    // Parse field 3: scale factor.

    double scale = std::stof(fields[3]); // positive: big endian, negative: little endian
    if (!(scale > 0.0 || scale < 0.0))
    {
        fclose(f);
        return "Invalid PFM scale factor!";
    }

    // Allocate pixels.

    Vec3f* pixels = new Vec3f[width * height];
    if (!pixels)
    {
        fclose(f);
        return "Out of memory!";
    }

    // Read raster data in bottom-up order.

    unsigned int* pixelsU32 = (unsigned int*)pixels;
    for (int y = height - 1; y >= 0; y--)
    {
        if (fread(pixelsU32 + y * width * numChannels, numChannels * sizeof(unsigned int), width, f) != (size_t)width)
        {
            delete[] pixels;
            fclose(f);
            return "Failed to read file!";
        }
    }

    // Grayscale => expand to RGB.

    if (numChannels == 1)
        for (int i = width * height - 1; i >= 0; i--)
            pixelsU32[i * 3 + 0] = pixelsU32[i * 3 + 1] = pixelsU32[i * 3 + 2] = pixelsU32[i];

    // Wrong endianess => reverse bytes.

    if ((scale > 0.0f) != (*(const unsigned int*)"\x01\x02\x03\x04" == 0x01020304))
    {
        for (int i = width * height * 3 - 1; i >= 0; i--)
        {
            unsigned int v = pixelsU32[i];
            pixelsU32[i] = (v >> 24) | ((v >> 8) & 0x0000FF00u) | ((v << 8) & 0x00FF0000u) | (v << 24);
        }
    }

    // Success.

    fclose(f);
    *pixelsOut = pixels;
    *widthOut = width;
    *heightOut = height;
    return NULL;
}

//------------------------------------------------------------------------

const char* poisson::exportPfmImage(const char* path, const Vec3f* pixels, int width, int height)
{
    assert(path && pixels);
    assert(width > 0 && height > 0);
    
    // Open file.

    FILE* f = NULL;
    if (fopen_s(&f, path, "wb") != 0)
        return "Failed to open file!";

    // Write header.

    bool bigEndian = (*(const unsigned int*)"\x01\x02\x03\x04" == 0x01020304);
    std::string header = sprintf("PF\n%d %d\n%g\n", width, height, (bigEndian) ? 1.0f : -1.0f);
    if (fwrite(header.c_str(), sizeof(char), header.length(), f) != header.length())
    {
        fclose(f);
        return "Failed to write file!";
    }

    // Write raster data in bottom-up order.

    for (int y = height - 1; y >= 0; y--)
    {
        if (fwrite(pixels + y * width, sizeof(Vec3f), width, f) != (size_t)width)
        {
            fclose(f);
            return "Failed to write file!";
        }
    }

    // Success.

    fclose(f);
    return NULL;
}

//------------------------------------------------------------------------
