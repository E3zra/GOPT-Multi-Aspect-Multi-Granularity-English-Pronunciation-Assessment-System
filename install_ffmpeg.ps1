# 快速安装 ffmpeg 脚本
# 自动下载并配置 ffmpeg

Write-Host "正在下载 ffmpeg..." -ForegroundColor Green

# 创建临时目录
$ffmpegDir = "C:\ffmpeg"
if (!(Test-Path $ffmpegDir)) {
    New-Item -ItemType Directory -Path $ffmpegDir | Out-Null
}

# 下载 ffmpeg (使用 GitHub 镜像)
$ffmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
$zipPath = "$env:TEMP\ffmpeg.zip"

Write-Host "下载 ffmpeg 到 $zipPath..." -ForegroundColor Yellow

try {
    Invoke-WebRequest -Uri $ffmpegUrl -OutFile $zipPath -UseBasicParsing
    Write-Host "下载完成！" -ForegroundColor Green
    
    Write-Host "解压中..." -ForegroundColor Yellow
    Expand-Archive -Path $zipPath -DestinationPath $env:TEMP -Force
    
    # 查找 bin 目录
    $binPath = Get-ChildItem -Path $env:TEMP -Filter "ffmpeg-master-*" -Directory | Select-Object -First 1
    $binFullPath = Join-Path $binPath.FullName "bin"
    
    Write-Host "复制 ffmpeg 到 $ffmpegDir..." -ForegroundColor Yellow
    Copy-Item -Path "$binFullPath\*" -Destination $ffmpegDir -Force
    
    # 添加到 PATH
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($currentPath -notlike "*$ffmpegDir*") {
        Write-Host "添加 ffmpeg 到 PATH..." -ForegroundColor Yellow
        [Environment]::SetEnvironmentVariable("Path", "$currentPath;$ffmpegDir", "User")
        $env:Path = "$env:Path;$ffmpegDir"
    }
    
    Write-Host "清理临时文件..." -ForegroundColor Yellow
    Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
    Remove-Item $binPath.FullName -Recurse -Force -ErrorAction SilentlyContinue
    
    Write-Host "`n安装完成！" -ForegroundColor Green
    Write-Host "ffmpeg 已安装到: $ffmpegDir" -ForegroundColor Cyan
    Write-Host "请重启命令行窗口或运行: `$env:Path = `"$env:Path;$ffmpegDir`"" -ForegroundColor Yellow
    
    # 测试
    Write-Host "`n测试 ffmpeg..." -ForegroundColor Green
    & "$ffmpegDir\ffmpeg.exe" -version | Select-Object -First 1
    
} catch {
    Write-Host "安装失败: $_" -ForegroundColor Red
    Write-Host "请手动下载: $ffmpegUrl" -ForegroundColor Yellow
}

