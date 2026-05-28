# Quick ffmpeg installation script
# Automatically downloads and configures ffmpeg

Write-Host "Downloading ffmpeg..." -ForegroundColor Green

# Create target directory
$ffmpegDir = "C:\ffmpeg"
if (!(Test-Path $ffmpegDir)) {
    New-Item -ItemType Directory -Path $ffmpegDir | Out-Null
}

# Download ffmpeg (using GitHub mirror)
$ffmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
$zipPath = "$env:TEMP\ffmpeg.zip"

Write-Host "Downloading ffmpeg to $zipPath..." -ForegroundColor Yellow

try {
    Invoke-WebRequest -Uri $ffmpegUrl -OutFile $zipPath -UseBasicParsing
    Write-Host "Download complete!" -ForegroundColor Green
    
    Write-Host "Extracting..." -ForegroundColor Yellow
    Expand-Archive -Path $zipPath -DestinationPath $env:TEMP -Force
    
    # Find the bin directory
    $binPath = Get-ChildItem -Path $env:TEMP -Filter "ffmpeg-master-*" -Directory | Select-Object -First 1
    $binFullPath = Join-Path $binPath.FullName "bin"
    
    Write-Host "Copying ffmpeg to $ffmpegDir..." -ForegroundColor Yellow
    Copy-Item -Path "$binFullPath\*" -Destination $ffmpegDir -Force
    
    # Add to PATH
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($currentPath -notlike "*$ffmpegDir*") {
        Write-Host "Adding ffmpeg to PATH..." -ForegroundColor Yellow
        [Environment]::SetEnvironmentVariable("Path", "$currentPath;$ffmpegDir", "User")
        $env:Path = "$env:Path;$ffmpegDir"
    }
    
    Write-Host "Cleaning up temporary files..." -ForegroundColor Yellow
    Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
    Remove-Item $binPath.FullName -Recurse -Force -ErrorAction SilentlyContinue
    
    Write-Host "`nInstallation complete!" -ForegroundColor Green
    Write-Host "ffmpeg installed to: $ffmpegDir" -ForegroundColor Cyan
    Write-Host "Please restart your command prompt or run: `$env:Path = `"$env:Path;$ffmpegDir`"" -ForegroundColor Yellow
    
    # Test
    Write-Host "`nTesting ffmpeg..." -ForegroundColor Green
    & "$ffmpegDir\ffmpeg.exe" -version | Select-Object -First 1
    
} catch {
    Write-Host "Installation failed: $_" -ForegroundColor Red
    Write-Host "Please download manually: $ffmpegUrl" -ForegroundColor Yellow
}