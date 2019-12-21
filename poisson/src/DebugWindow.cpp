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

#include "DebugWindow.hpp"

using namespace poisson;

//------------------------------------------------------------------------

DebugWindow::DebugWindow(void)
:   m_width     (1),
    m_height    (1),

    m_hwnd      (NULL),
    m_hdc       (NULL),
    m_memdc     (NULL),

    m_dib       (NULL),
    m_dibPtr    (NULL)
{
}

//------------------------------------------------------------------------

DebugWindow::~DebugWindow(void)
{
    deinitDIB();
    deinitWindow();
}

//------------------------------------------------------------------------

void DebugWindow::setTitle(const char* title)
{
    assert(title);
    initWindow();
    if (!SetWindowText(m_hwnd, title))
        fail("SetWindowText() failed!");
    processMessages();
}

//------------------------------------------------------------------------

void DebugWindow::setSize(int width, int height)
{
    assert(width > 0 && height > 0);
    deinitDIB();
    m_width = width;
    m_height = height;

    // Window already initialized => resize.

    if (m_hwnd)
    {
        RECT rc;
        rc.left     = 0;
        rc.top      = 0;
        rc.right    = m_width;
        rc.bottom   = m_height;
        AdjustWindowRect(&rc, GetWindowLong(m_hwnd, GWL_STYLE), (GetMenu(m_hwnd) != NULL));

        SetWindowPos(m_hwnd, NULL, 0, 0, rc.right - rc.left, rc.bottom - rc.top,
            SWP_NOACTIVATE | SWP_NOCOPYBITS | SWP_NOMOVE | SWP_NOZORDER);

        processMessages();
    }
}

//------------------------------------------------------------------------

unsigned int* DebugWindow::getPixelPtr(void)
{
    initDIB();
    return m_dibPtr;
}

//------------------------------------------------------------------------

void DebugWindow::display(void)
{
    // Make the window visible.

    initWindow();
    ShowWindow(m_hwnd, SW_SHOW);
    processMessages();

    // Copy DIB to screen.
    // Note: BitBlt() may occassionally fail when using remote desktop.

    if (m_dib)
        BitBlt(m_hdc, 0, 0, m_width, m_height, m_memdc, 0, 0, SRCCOPY);
}

//------------------------------------------------------------------------

void DebugWindow::hide(void)
{
    if (m_hwnd)
    {
        ShowWindow(m_hwnd, SW_HIDE);
        processMessages();
    }
}

//------------------------------------------------------------------------

void DebugWindow::initWindow(void)
{
    if (m_hwnd)
        return;

    // Register window class.

    WNDCLASS wc;
    wc.style            = 0;
    wc.lpfnWndProc      = DefWindowProc;
    wc.cbClsExtra       = 0;
    wc.cbWndExtra       = 0;
    wc.hInstance        = (HINSTANCE)GetModuleHandle(NULL);
    wc.hIcon            = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor          = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground    = CreateSolidBrush(RGB(0, 0, 0));
    wc.lpszMenuName     = NULL;
    wc.lpszClassName    = "DebugWindowClass";
    RegisterClass(&wc);

    // Create window.

    m_hwnd = CreateWindow(
        wc.lpszClassName,
        "Debug window",
        WS_OVERLAPPEDWINDOW,
        0, 0, m_width, m_height,
        NULL,
        NULL,
        (HINSTANCE)GetModuleHandle(NULL),
        NULL);

    if (!m_hwnd)
        fail("CreateWindow() failed!");

    // Get device contexts.

    m_hdc = GetDC(m_hwnd);
    if (!m_hdc)
        fail("GetDC() failed!");

    m_memdc = CreateCompatibleDC(m_hdc);
    if (!m_memdc)
        fail("CreateCompatibleDC() failed!");

    // Process pending window messages.

    processMessages();
}

//------------------------------------------------------------------------

void DebugWindow::deinitWindow(void)
{
    if (m_memdc)
        DeleteObject(m_memdc);

    if (m_hdc)
        ReleaseDC(m_hwnd, m_hdc);

    if (m_hwnd)
        DestroyWindow(m_hwnd);

    m_memdc = NULL;
    m_hwnd = NULL;
}

//------------------------------------------------------------------------

void DebugWindow::initDIB(void)
{
    if (m_dib)
        return;
    initWindow();

    // Create DIB.

    char bmi[sizeof(BITMAPINFOHEADER) + 3 * sizeof(DWORD)];
    BITMAPINFOHEADER* bmih  = (BITMAPINFOHEADER*)bmi;
    DWORD* masks            = (DWORD*)(bmih + 1);

    bmih->biSize            = sizeof(BITMAPINFOHEADER);
    bmih->biWidth           = m_width;
    bmih->biHeight          = -m_height;
    bmih->biPlanes          = 1;
    bmih->biBitCount        = 32;
    bmih->biCompression     = BI_BITFIELDS;
    bmih->biSizeImage       = 0;
    bmih->biXPelsPerMeter   = 0;
    bmih->biYPelsPerMeter   = 0;
    bmih->biClrUsed         = 0;
    bmih->biClrImportant    = 0;
    masks[0]                = 0x000000FF;
    masks[1]                = 0x0000FF00;
    masks[2]                = 0x00FF0000;

    m_dib = CreateDIBSection(m_memdc, (BITMAPINFO*)bmi, DIB_RGB_COLORS, (void**)&m_dibPtr, NULL, 0);
    if (!m_dib)
        fail("CreateDIBSection() failed!");

    // Clear DIB.

    memset(m_dibPtr, 0, m_width * m_height * sizeof(unsigned int));

    // Bind to the device context.

    if (!SelectObject(m_memdc, m_dib))
        fail("SelectObject() failed!");
}

//------------------------------------------------------------------------

void DebugWindow::deinitDIB(void)
{
    if (m_dib)
        DeleteObject(m_dib);

    m_dib = NULL;
    m_dibPtr = NULL;
}

//------------------------------------------------------------------------

void DebugWindow::processMessages(void)
{
    MSG msg;
    while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

//------------------------------------------------------------------------
