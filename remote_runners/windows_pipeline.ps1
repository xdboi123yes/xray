[CmdletBinding()]
param(
    [string]$RepoUrl = "https://github.com/xdboi123yes/xray.git",
    [string]$Branch = "main",
    [string]$Workspace = "$env:USERPROFILE\XRay-Production",
    [switch]$Worker,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

function Find-Python {
    if (Get-Command py -ErrorAction SilentlyContinue) { return @("py", "-3.11") }
    if (Get-Command python -ErrorAction SilentlyContinue) { return @("python") }
    if (Get-Command python3 -ErrorAction SilentlyContinue) { return @("python3") }
    return $null
}

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments, [string]$WorkingDirectory = $PWD.Path)
    Write-Host ("`n> " + $Exe + " " + ($Arguments -join " ")) -ForegroundColor Cyan
    Push-Location $WorkingDirectory
    try {
        & $Exe @Arguments
        if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE: $Exe" }
    } finally { Pop-Location }
}

function Install-WingetPackage {
    param([string]$Id, [string]$Label)
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "winget is unavailable. Install Microsoft App Installer, then run this launcher again."
    }
    Write-Host "Installing $Label..." -ForegroundColor Yellow
    & winget install --id $Id --exact --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "winget could not install $Label ($Id)." }
    Refresh-Path
}

function Enable-StayAwake {
    Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class Awake {
  [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
  public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
    [void][Awake]::SetThreadExecutionState(0x80000003) # continuous + system + display
}

function Disable-StayAwake {
    if ("Awake" -as [type]) { [void][Awake]::SetThreadExecutionState(0x80000000) }
}

function Start-Worker {
    New-Item -ItemType Directory -Force -Path $Workspace | Out-Null
    $repo = Join-Path $Workspace "repo"
    $venv = Join-Path $Workspace "venv"
    $logs = Join-Path $Workspace "logs"
    New-Item -ItemType Directory -Force -Path $logs, (Join-Path $Workspace "runtime"), (Join-Path $Workspace "artifacts") | Out-Null

    Enable-StayAwake
    try {
        Write-Host "XRAY WINDOWS PRODUCTION CONTROL" -ForegroundColor Black -BackgroundColor Cyan
        Write-Host "Workspace: $Workspace"

        if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
            Install-WingetPackage "Git.Git" "Git"
        }
        $pythonCmd = Find-Python
        if (-not $pythonCmd) {
            Install-WingetPackage "Python.Python.3.11" "Python 3.11"
            $pythonCmd = Find-Python
        }
        if (-not $pythonCmd) { throw "Python installation completed but Python is not visible. Restart Windows and try again." }

        if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue) -and
            -not (Get-Command tectonic -ErrorAction SilentlyContinue)) {
            Install-WingetPackage "MiKTeX.MiKTeX" "MiKTeX"
        }

        if (-not (Test-Path (Join-Path $repo ".git"))) {
            Invoke-Checked "git" @("clone", "--branch", $Branch, "--single-branch", $RepoUrl, $repo) $Workspace
        } else {
            $dirty = & git -C $repo status --porcelain
            if ($dirty) { throw "The server repository contains local changes. Preserve them before continuing: $repo" }
            Invoke-Checked "git" @("-C", $repo, "fetch", "origin", $Branch)
            Invoke-Checked "git" @("-C", $repo, "checkout", $Branch)
            Invoke-Checked "git" @("-C", $repo, "pull", "--ff-only", "origin", $Branch)
        }

        if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
            $exe = $pythonCmd[0]; $prefix = @()
            if ($pythonCmd.Count -gt 1) { $prefix = $pythonCmd[1..($pythonCmd.Count - 1)] }
            Invoke-Checked $exe ($prefix + @("-m", "venv", $venv)) $Workspace
        }

        $python = Join-Path $venv "Scripts\python.exe"
        Invoke-Checked $python @("-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools") $repo
        Invoke-Checked $python @("-m", "pip", "install", "-r", "requirements.txt", "-r", "requirements-training.txt") $repo
        Invoke-Checked $python @("-m", "pip", "install", "rich", "psutil", "jupyter", "nbconvert", "ipykernel") $repo
        Invoke-Checked $python @("-m", "ipykernel", "install", "--user", "--name", "xray-production", "--display-name", "XRay Production") $repo

        $env:XRAY_PROJECT_ROOT = $repo
        $env:XRAY_RUNTIME_ROOT = Join-Path $Workspace "runtime"
        $env:XRAY_WORKSPACE = $Workspace
        $env:PYTHONUNBUFFERED = "1"

        $controllerArgs = @((Join-Path $repo "scripts\university_pipeline.py"))
        if ($PreflightOnly) { $controllerArgs += "--preflight-only" }
        Invoke-Checked $python $controllerArgs $repo

        Set-Content -Path (Join-Path $Workspace "COMPLETE.txt") -Value (Get-Date).ToString("o")
        Write-Host "`nCOMPLETE — presentation bundle is ready under $Workspace\artifacts" -ForegroundColor Green
    } finally {
        Disable-StayAwake
    }
}

function Show-ControlPanel {
    Add-Type -AssemblyName PresentationFramework, PresentationCore, WindowsBase
    New-Item -ItemType Directory -Force -Path $Workspace, (Join-Path $Workspace "logs") | Out-Null
    $stdout = Join-Path $Workspace "logs\windows-worker.log"
    $stderr = Join-Path $Workspace "logs\windows-worker-error.log"

    [xml]$xaml = @"
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        Title="XRay Production Control" Height="690" Width="1040"
        WindowStartupLocation="CenterScreen" Background="#07100F" Foreground="#D9F7EE">
  <Grid Margin="28">
    <Grid.RowDefinitions><RowDefinition Height="Auto"/><RowDefinition Height="Auto"/><RowDefinition Height="*"/><RowDefinition Height="Auto"/></Grid.RowDefinitions>
    <Border Grid.Row="0" BorderBrush="#22D3A7" BorderThickness="0,0,0,2" Padding="0,0,0,18">
      <StackPanel>
        <TextBlock Text="XRAY / PRODUCTION CONTROL" FontFamily="Bahnschrift Condensed" FontSize="34" FontWeight="Bold" Foreground="#58F5C6"/>
        <TextBlock Text="PATIENT-DISJOINT TRAINING • VERIFIED OUTPUTS • FINAL MANUSCRIPT" FontFamily="Consolas" FontSize="12" Foreground="#72A99A" Margin="2,6,0,0"/>
      </StackPanel>
    </Border>
    <Grid Grid.Row="1" Margin="0,18,0,16">
      <Grid.ColumnDefinitions><ColumnDefinition Width="*"/><ColumnDefinition Width="Auto"/></Grid.ColumnDefinitions>
      <StackPanel>
        <TextBlock Name="StatusText" Text="READY TO START" FontFamily="Bahnschrift" FontSize="18" FontWeight="SemiBold"/>
        <TextBlock Name="DetailText" Text="The PC will stay awake while the pipeline is active." Foreground="#72A99A" Margin="0,5,0,0"/>
      </StackPanel>
      <ProgressBar Name="Pulse" Grid.Column="1" Width="280" Height="8" IsIndeterminate="False" Foreground="#22D3A7" Background="#17342D"/>
    </Grid>
    <Border Grid.Row="2" Background="#030706" BorderBrush="#17342D" BorderThickness="1" Padding="16">
      <TextBox Name="LogBox" IsReadOnly="True" TextWrapping="NoWrap" VerticalScrollBarVisibility="Auto" HorizontalScrollBarVisibility="Auto"
               Background="#030706" Foreground="#9FE8D3" BorderThickness="0" FontFamily="Cascadia Mono,Consolas" FontSize="12"/>
    </Border>
    <Grid Grid.Row="3" Margin="0,18,0,0">
      <Grid.ColumnDefinitions><ColumnDefinition Width="Auto"/><ColumnDefinition Width="Auto"/><ColumnDefinition Width="*"/><ColumnDefinition Width="Auto"/></Grid.ColumnDefinitions>
      <Button Name="StartButton" Content="START FULL PIPELINE" Padding="22,11" Background="#22D3A7" Foreground="#04110D" FontWeight="Bold" BorderThickness="0"/>
      <Button Name="PreflightButton" Grid.Column="1" Content="PREFLIGHT ONLY" Margin="12,0,0,0" Padding="18,11" Background="#17342D" Foreground="#D9F7EE" BorderThickness="0"/>
      <Button Name="FolderButton" Grid.Column="3" Content="OPEN OUTPUT FOLDER" Padding="18,11" Background="#17342D" Foreground="#D9F7EE" BorderThickness="0"/>
    </Grid>
  </Grid>
</Window>
"@
    $reader = New-Object System.Xml.XmlNodeReader $xaml
    $window = [Windows.Markup.XamlReader]::Load($reader)
    $start = $window.FindName("StartButton")
    $preflight = $window.FindName("PreflightButton")
    $folder = $window.FindName("FolderButton")
    $status = $window.FindName("StatusText")
    $detail = $window.FindName("DetailText")
    $pulse = $window.FindName("Pulse")
    $logBox = $window.FindName("LogBox")
    $scriptPath = $MyInvocation.MyCommand.Path
    $script:process = $null

    $launch = {
        param([bool]$OnlyPreflight)
        $kaggleTarget = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
        if (-not (Test-Path $kaggleTarget)) {
            Add-Type -AssemblyName System.Windows.Forms
            $picker = New-Object System.Windows.Forms.OpenFileDialog
            $picker.Title = "Select your Kaggle API kaggle.json"
            $picker.Filter = "Kaggle credentials (kaggle.json)|kaggle.json|JSON files (*.json)|*.json"
            if ($picker.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
                [System.Windows.MessageBox]::Show("Kaggle credentials are required to download NIH and CheXpert data.", "Credentials required") | Out-Null
                return
            }
            New-Item -ItemType Directory -Force -Path (Split-Path $kaggleTarget) | Out-Null
            Copy-Item $picker.FileName $kaggleTarget -Force
        }
        Remove-Item $stdout, $stderr -Force -ErrorAction SilentlyContinue
        $argLine = "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -Worker -RepoUrl `"$RepoUrl`" -Branch `"$Branch`" -Workspace `"$Workspace`""
        if ($OnlyPreflight) { $argLine += " -PreflightOnly" }
        $script:process = Start-Process powershell.exe -ArgumentList $argLine -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $status.Text = if ($OnlyPreflight) { "PREFLIGHT RUNNING" } else { "PRODUCTION RUNNING" }
        $detail.Text = "AnyDesk may disconnect; the local worker continues while Windows remains powered on."
        $pulse.IsIndeterminate = $true
        $start.IsEnabled = $false; $preflight.IsEnabled = $false
    }
    $start.Add_Click({ & $launch $false })
    $preflight.Add_Click({ & $launch $true })
    $folder.Add_Click({ New-Item -ItemType Directory -Force -Path (Join-Path $Workspace "artifacts") | Out-Null; Start-Process explorer.exe (Join-Path $Workspace "artifacts") })

    $timer = New-Object Windows.Threading.DispatcherTimer
    $timer.Interval = [TimeSpan]::FromSeconds(1)
    $timer.Add_Tick({
        $content = ""
        if (Test-Path $stdout) { $content += (Get-Content $stdout -Raw -ErrorAction SilentlyContinue) }
        if (Test-Path $stderr) { $content += "`r`n" + (Get-Content $stderr -Raw -ErrorAction SilentlyContinue) }
        $logBox.Text = $content
        $logBox.ScrollToEnd()
        if ($script:process -and $script:process.HasExited) {
            $pulse.IsIndeterminate = $false
            if ($script:process.ExitCode -eq 0) {
                $status.Text = "COMPLETE / PRESENTATION BUNDLE READY"
                $status.Foreground = "#58F5C6"
                $detail.Text = "$Workspace\artifacts"
            } else {
                $status.Text = "STOPPED / REVIEW LOG"
                $status.Foreground = "#FF866E"
                $detail.Text = "Exit code $($script:process.ExitCode)"
            }
            $start.IsEnabled = $true; $preflight.IsEnabled = $true
            $script:process = $null
        }
    })
    $timer.Start()
    [void]$window.ShowDialog()
}

if ($Worker) { Start-Worker } else { Show-ControlPanel }
