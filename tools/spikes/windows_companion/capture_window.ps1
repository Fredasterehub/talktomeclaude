param(
    [Parameter(Mandatory = $true)][Int64]$Hwnd,
    [Parameter(Mandatory = $true)][string]$OutputPath
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$native = @'
using System;
using System.Runtime.InteropServices;
public static class SpikeScreenshotNative {
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }
    [DllImport("user32.dll", SetLastError=true)]
    public static extern bool GetWindowRect(IntPtr hwnd, out RECT rect);
    [DllImport("user32.dll", SetLastError=true)]
    public static extern bool PrintWindow(IntPtr hwnd, IntPtr hdc, uint flags);
}
'@
Add-Type -TypeDefinition $native -Language CSharp
$rect = New-Object SpikeScreenshotNative+RECT
if (-not [SpikeScreenshotNative]::GetWindowRect([IntPtr]$Hwnd, [ref]$rect)) {
    throw "GetWindowRect failed"
}
$width = $rect.Right - $rect.Left
$height = $rect.Bottom - $rect.Top
if ($width -le 0 -or $height -le 0 -or $width -gt 2000 -or $height -gt 1200) {
    throw "Refusing unexpected screenshot bounds ${width}x${height}"
}
$directory = Split-Path -Parent $OutputPath
[System.IO.Directory]::CreateDirectory($directory) | Out-Null
$bitmap = [System.Drawing.Bitmap]::new($width, $height)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
try {
    $hdc = $graphics.GetHdc()
    try {
        if (-not [SpikeScreenshotNative]::PrintWindow([IntPtr]$Hwnd, $hdc, 2)) { throw 'PrintWindow failed' }
    } finally { $graphics.ReleaseHdc($hdc) }
    $bitmap.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
} finally {
    $graphics.Dispose()
    $bitmap.Dispose()
}
@{ width = $width; height = $height; path = $OutputPath } | ConvertTo-Json -Compress
