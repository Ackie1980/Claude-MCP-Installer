<#
.SYNOPSIS
    Claude MCP Installer - GUI Version
.DESCRIPTION
    A Windows Forms GUI wizard for installing Claude Code CLI and configuring MCP servers.
.NOTES
    Version: 2.0.0
    Author: Quality By Design NV
#>

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# ============================================================================
# Import Core Functions from Main Script
# ============================================================================

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$scriptPath\Install-ClaudeMCP.ps1" -NoRun

# ============================================================================
# Wrapper Functions for Prerequisite Checks
# ============================================================================
# These wrap the Get-*Version functions from the main script

function Test-NodeJS {
    $version = Get-NodeVersion
    return ($null -ne $version)
}

function Test-Python {
    $version = Get-PythonVersion
    return ($null -ne $version)
}

function Test-UV {
    $version = Get-UVVersion
    return ($null -ne $version)
}

function Test-ClaudeCode {
    $version = Get-ClaudeVersion
    return ($null -ne $version)
}

# ============================================================================
# GUI Configuration
# ============================================================================

$script:GuiConfig = @{
    WindowWidth = 700
    WindowHeight = 550
    PrimaryColor = [System.Drawing.Color]::FromArgb(45, 45, 48)
    SecondaryColor = [System.Drawing.Color]::FromArgb(62, 62, 66)
    AccentColor = [System.Drawing.Color]::FromArgb(0, 122, 204)
    TextColor = [System.Drawing.Color]::White
    SuccessColor = [System.Drawing.Color]::FromArgb(78, 201, 176)
    WarningColor = [System.Drawing.Color]::FromArgb(255, 204, 0)
    ErrorColor = [System.Drawing.Color]::FromArgb(255, 99, 71)
    FontFamily = "Segoe UI"
    FontSize = 10
}

$script:CurrentPage = 0
$script:SelectedServers = @()
$script:ServerConfigs = @{}

# ============================================================================
# GUI Helper Functions
# ============================================================================

function New-StyledButton {
    param(
        [string]$Text,
        [int]$Width = 100,
        [int]$Height = 35,
        [System.Drawing.Color]$BackColor = $script:GuiConfig.AccentColor,
        [bool]$Enabled = $true
    )

    $button = New-Object System.Windows.Forms.Button
    $button.Text = $Text
    $button.Width = $Width
    $button.Height = $Height
    $button.FlatStyle = [System.Windows.Forms.FlatStyle]::Flat
    $button.FlatAppearance.BorderSize = 0
    $button.BackColor = $BackColor
    $button.ForeColor = $script:GuiConfig.TextColor
    $button.Font = New-Object System.Drawing.Font($script:GuiConfig.FontFamily, 10, [System.Drawing.FontStyle]::Regular)
    $button.Cursor = [System.Windows.Forms.Cursors]::Hand
    $button.Enabled = $Enabled

    return $button
}

function New-StyledLabel {
    param(
        [string]$Text,
        [int]$FontSize = $script:GuiConfig.FontSize,
        [System.Drawing.FontStyle]$FontStyle = [System.Drawing.FontStyle]::Regular,
        [System.Drawing.Color]$ForeColor = $script:GuiConfig.TextColor
    )

    $label = New-Object System.Windows.Forms.Label
    $label.Text = $Text
    $label.Font = New-Object System.Drawing.Font($script:GuiConfig.FontFamily, $FontSize, $FontStyle)
    $label.ForeColor = $ForeColor
    $label.AutoSize = $true
    $label.BackColor = [System.Drawing.Color]::Transparent

    return $label
}

function New-StyledCheckBox {
    param(
        [string]$Text,
        [string]$Tag = ""
    )

    $checkbox = New-Object System.Windows.Forms.CheckBox
    $checkbox.Text = $Text
    $checkbox.Font = New-Object System.Drawing.Font($script:GuiConfig.FontFamily, 10, [System.Drawing.FontStyle]::Regular)
    $checkbox.ForeColor = $script:GuiConfig.TextColor
    $checkbox.BackColor = [System.Drawing.Color]::Transparent
    $checkbox.AutoSize = $true
    $checkbox.Tag = $Tag
    $checkbox.Cursor = [System.Windows.Forms.Cursors]::Hand

    return $checkbox
}

# ============================================================================
# Page Creation Functions
# ============================================================================

function New-WelcomePage {
    $panel = New-Object System.Windows.Forms.Panel
    $panel.Dock = [System.Windows.Forms.DockStyle]::Fill
    $panel.BackColor = $script:GuiConfig.PrimaryColor

    # Title
    $title = New-StyledLabel -Text "Claude MCP Installer" -FontSize 24 -FontStyle ([System.Drawing.FontStyle]::Bold)
    $title.Location = New-Object System.Drawing.Point(50, 40)
    $panel.Controls.Add($title)

    # Subtitle
    $subtitle = New-StyledLabel -Text "Version 2.0.0" -FontSize 12 -ForeColor ([System.Drawing.Color]::Gray)
    $subtitle.Location = New-Object System.Drawing.Point(50, 85)
    $panel.Controls.Add($subtitle)

    # Description
    $desc = New-Object System.Windows.Forms.Label
    $desc.Text = @"
Welcome to the Claude MCP Installer!

This wizard will help you:

    - Install Claude Code CLI (command-line interface)
    - Configure MCP (Model Context Protocol) servers
    - Set up integrations for Microsoft 365, Excel, PowerBI, and more

Before you begin:
    - Ensure you have an internet connection
    - Close Claude Desktop if it's running
    - Some components may require administrator privileges

Click 'Next' to check prerequisites and begin the installation.
"@
    $desc.Font = New-Object System.Drawing.Font($script:GuiConfig.FontFamily, 10, [System.Drawing.FontStyle]::Regular)
    $desc.ForeColor = $script:GuiConfig.TextColor
    $desc.BackColor = [System.Drawing.Color]::Transparent
    $desc.Location = New-Object System.Drawing.Point(50, 130)
    $desc.Size = New-Object System.Drawing.Size(580, 280)
    $panel.Controls.Add($desc)

    # Author
    $author = New-StyledLabel -Text "By Quality By Design NV" -FontSize 9 -ForeColor ([System.Drawing.Color]::Gray)
    $author.Location = New-Object System.Drawing.Point(50, 420)
    $panel.Controls.Add($author)

    return $panel
}

function New-PrerequisitesPage {
    $panel = New-Object System.Windows.Forms.Panel
    $panel.Dock = [System.Windows.Forms.DockStyle]::Fill
    $panel.BackColor = $script:GuiConfig.PrimaryColor

    # Title
    $title = New-StyledLabel -Text "Prerequisites Check" -FontSize 18 -FontStyle ([System.Drawing.FontStyle]::Bold)
    $title.Location = New-Object System.Drawing.Point(50, 30)
    $panel.Controls.Add($title)

    # Description
    $desc = New-StyledLabel -Text "Checking required components..." -FontSize 10
    $desc.Location = New-Object System.Drawing.Point(50, 70)
    $panel.Controls.Add($desc)

    # Status panel
    $statusPanel = New-Object System.Windows.Forms.Panel
    $statusPanel.Location = New-Object System.Drawing.Point(50, 110)
    $statusPanel.Size = New-Object System.Drawing.Size(580, 250)
    $statusPanel.BackColor = $script:GuiConfig.SecondaryColor
    $statusPanel.Name = "StatusPanel"
    $panel.Controls.Add($statusPanel)

    # Install button
    $installBtn = New-StyledButton -Text "Install Missing" -Width 150 -BackColor ([System.Drawing.Color]::FromArgb(0, 150, 136))
    $installBtn.Location = New-Object System.Drawing.Point(50, 380)
    $installBtn.Name = "InstallPrereqBtn"
    $installBtn.Visible = $false
    $panel.Controls.Add($installBtn)

    # Refresh button
    $refreshBtn = New-StyledButton -Text "Refresh" -Width 100 -BackColor $script:GuiConfig.SecondaryColor
    $refreshBtn.Location = New-Object System.Drawing.Point(210, 380)
    $refreshBtn.Name = "RefreshBtn"
    $panel.Controls.Add($refreshBtn)

    return $panel
}

function Update-PrerequisitesStatus {
    param($Panel)

    $statusPanel = $Panel.Controls["StatusPanel"]
    $statusPanel.Controls.Clear()

    # Check components
    $components = @(
        @{ Name = "Node.js"; Check = { Test-NodeJS }; Install = { Install-NodeJS } },
        @{ Name = "Python 3"; Check = { Test-Python }; Install = { Install-Python } },
        @{ Name = "UV (Python Package Manager)"; Check = { Test-UV }; Install = { Install-UV } },
        @{ Name = "Claude Code CLI"; Check = { Test-ClaudeCode }; Install = { Install-ClaudeCode } }
    )

    $yPos = 15
    $missingCount = 0

    foreach ($component in $components) {
        $isInstalled = & $component.Check

        $statusLabel = New-Object System.Windows.Forms.Label
        $statusLabel.Location = New-Object System.Drawing.Point(15, $yPos)
        $statusLabel.Size = New-Object System.Drawing.Size(550, 45)
        $statusLabel.Font = New-Object System.Drawing.Font($script:GuiConfig.FontFamily, 11, [System.Drawing.FontStyle]::Regular)
        $statusLabel.BackColor = [System.Drawing.Color]::Transparent

        if ($isInstalled) {
            $statusLabel.Text = "[Installed]  $($component.Name)"
            $statusLabel.ForeColor = $script:GuiConfig.SuccessColor
        } else {
            $statusLabel.Text = "[Missing]    $($component.Name)"
            $statusLabel.ForeColor = $script:GuiConfig.WarningColor
            $missingCount++
        }

        $statusPanel.Controls.Add($statusLabel)
        $yPos += 50
    }

    # Show/hide install button
    $installBtn = $Panel.Controls["InstallPrereqBtn"]
    $installBtn.Visible = ($missingCount -gt 0)

    return $missingCount
}

function New-ServerSelectionPage {
    $panel = New-Object System.Windows.Forms.Panel
    $panel.Dock = [System.Windows.Forms.DockStyle]::Fill
    $panel.BackColor = $script:GuiConfig.PrimaryColor

    # Title
    $title = New-StyledLabel -Text "Select MCP Servers" -FontSize 18 -FontStyle ([System.Drawing.FontStyle]::Bold)
    $title.Location = New-Object System.Drawing.Point(50, 30)
    $panel.Controls.Add($title)

    # Description
    $desc = New-StyledLabel -Text "Choose which MCP servers to install and configure:" -FontSize 10
    $desc.Location = New-Object System.Drawing.Point(50, 70)
    $panel.Controls.Add($desc)

    # Scrollable panel for checkboxes
    $scrollPanel = New-Object System.Windows.Forms.Panel
    $scrollPanel.Location = New-Object System.Drawing.Point(50, 110)
    $scrollPanel.Size = New-Object System.Drawing.Size(580, 300)
    $scrollPanel.BackColor = $script:GuiConfig.SecondaryColor
    $scrollPanel.AutoScroll = $true
    $scrollPanel.Name = "ServerScrollPanel"

    $yPos = 15
    foreach ($key in $MCPServers.Keys | Sort-Object) {
        $server = $MCPServers[$key]

        $checkPanel = New-Object System.Windows.Forms.Panel
        $checkPanel.Location = New-Object System.Drawing.Point(10, $yPos)
        $checkPanel.Size = New-Object System.Drawing.Size(540, 50)
        $checkPanel.BackColor = [System.Drawing.Color]::Transparent

        $checkbox = New-StyledCheckBox -Text $server.Name -Tag $key
        $checkbox.Location = New-Object System.Drawing.Point(5, 5)
        $checkbox.Name = "chk_$key"
        $checkPanel.Controls.Add($checkbox)

        # Prerequisites indicator
        if ($server.Prerequisites) {
            $prereqLabel = New-Object System.Windows.Forms.Label
            $prereqLabel.Text = "Requires: $($server.Prerequisites -join ', ')"
            $prereqLabel.Font = New-Object System.Drawing.Font($script:GuiConfig.FontFamily, 8, [System.Drawing.FontStyle]::Italic)
            $prereqLabel.ForeColor = [System.Drawing.Color]::Gray
            $prereqLabel.Location = New-Object System.Drawing.Point(25, 28)
            $prereqLabel.AutoSize = $true
            $prereqLabel.BackColor = [System.Drawing.Color]::Transparent
            $checkPanel.Controls.Add($prereqLabel)
        }

        $scrollPanel.Controls.Add($checkPanel)
        $yPos += 55
    }

    $panel.Controls.Add($scrollPanel)

    # Select All / Deselect All buttons
    $selectAllBtn = New-StyledButton -Text "Select All" -Width 100 -BackColor $script:GuiConfig.SecondaryColor
    $selectAllBtn.Location = New-Object System.Drawing.Point(50, 420)
    $selectAllBtn.Name = "SelectAllBtn"
    $panel.Controls.Add($selectAllBtn)

    $deselectAllBtn = New-StyledButton -Text "Deselect All" -Width 100 -BackColor $script:GuiConfig.SecondaryColor
    $deselectAllBtn.Location = New-Object System.Drawing.Point(160, 420)
    $deselectAllBtn.Name = "DeselectAllBtn"
    $panel.Controls.Add($deselectAllBtn)

    return $panel
}

function New-ConfigurationPage {
    $panel = New-Object System.Windows.Forms.Panel
    $panel.Dock = [System.Windows.Forms.DockStyle]::Fill
    $panel.BackColor = $script:GuiConfig.PrimaryColor

    # Title
    $title = New-StyledLabel -Text "Configuration" -FontSize 18 -FontStyle ([System.Drawing.FontStyle]::Bold)
    $title.Location = New-Object System.Drawing.Point(50, 30)
    $panel.Controls.Add($title)

    # Description
    $desc = New-StyledLabel -Text "Configure settings for selected MCP servers:" -FontSize 10
    $desc.Location = New-Object System.Drawing.Point(50, 70)
    $panel.Controls.Add($desc)

    # Scrollable config panel
    $scrollPanel = New-Object System.Windows.Forms.Panel
    $scrollPanel.Location = New-Object System.Drawing.Point(50, 110)
    $scrollPanel.Size = New-Object System.Drawing.Size(580, 320)
    $scrollPanel.BackColor = $script:GuiConfig.SecondaryColor
    $scrollPanel.AutoScroll = $true
    $scrollPanel.Name = "ConfigScrollPanel"
    $panel.Controls.Add($scrollPanel)

    return $panel
}

function Update-ConfigurationPage {
    param($Panel, $SelectedServers)

    $scrollPanel = $Panel.Controls["ConfigScrollPanel"]
    $scrollPanel.Controls.Clear()

    $yPos = 15

    foreach ($serverKey in $SelectedServers) {
        $server = $MCPServers[$serverKey]

        # Check if server requires configuration or has setup notes
        if ($server.RequiresPath -or $server.RequiresApiKey -or $server.PreInstallNote) {
            # Calculate panel height based on content
            $panelHeight = 35  # Base height for title
            $contentYPos = 35  # Y position for content after title

            if ($server.PreInstallNote) {
                $panelHeight += 100  # Add space for pre-install note
            }
            if ($server.RequiresPath -or $server.RequiresApiKey) {
                $panelHeight += 45  # Add space for input field
            }

            $configPanel = New-Object System.Windows.Forms.Panel
            $configPanel.Location = New-Object System.Drawing.Point(10, $yPos)
            $configPanel.Size = New-Object System.Drawing.Size(540, $panelHeight)
            $configPanel.BackColor = [System.Drawing.Color]::FromArgb(50, 50, 54)

            # Server name
            $nameLabel = New-StyledLabel -Text $server.Name -FontSize 11 -FontStyle ([System.Drawing.FontStyle]::Bold)
            $nameLabel.Location = New-Object System.Drawing.Point(10, 8)
            $configPanel.Controls.Add($nameLabel)

            # Show pre-install note if present
            if ($server.PreInstallNote) {
                $noteLabel = New-Object System.Windows.Forms.Label
                $noteLabel.Text = $server.PreInstallNote.Trim()
                $noteLabel.Location = New-Object System.Drawing.Point(10, $contentYPos)
                $noteLabel.Size = New-Object System.Drawing.Size(520, 90)
                $noteLabel.Font = New-Object System.Drawing.Font($script:GuiConfig.FontFamily, 8)
                $noteLabel.ForeColor = $script:GuiConfig.WarningColor
                $noteLabel.BackColor = [System.Drawing.Color]::Transparent
                $configPanel.Controls.Add($noteLabel)
                $contentYPos += 95  # Move down for any subsequent controls
            }

            if ($server.RequiresPath) {
                $pathLabel = New-StyledLabel -Text "Path:" -FontSize 9
                $pathLabel.Location = New-Object System.Drawing.Point(10, $contentYPos)
                $configPanel.Controls.Add($pathLabel)

                $pathBox = New-Object System.Windows.Forms.TextBox
                $pathBox.Location = New-Object System.Drawing.Point(50, ($contentYPos - 3))
                $pathBox.Size = New-Object System.Drawing.Size(400, 25)
                $pathBox.Font = New-Object System.Drawing.Font($script:GuiConfig.FontFamily, 9)
                $pathBox.Name = "path_$serverKey"
                $pathBox.Text = $server.DefaultPath
                $configPanel.Controls.Add($pathBox)

                $browseBtn = New-StyledButton -Text "..." -Width 30 -Height 25 -BackColor $script:GuiConfig.SecondaryColor
                $browseBtn.Location = New-Object System.Drawing.Point(455, ($contentYPos - 3))
                $browseBtn.Tag = "path_$serverKey"
                $browseBtn.Add_Click({
                    $folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog
                    if ($folderBrowser.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
                        $textBox = $this.Parent.Controls[$this.Tag]
                        $textBox.Text = $folderBrowser.SelectedPath
                    }
                })
                $configPanel.Controls.Add($browseBtn)
            }

            if ($server.RequiresApiKey) {
                $apiLabel = New-StyledLabel -Text "API Key:" -FontSize 9
                $apiLabel.Location = New-Object System.Drawing.Point(10, $contentYPos)
                $configPanel.Controls.Add($apiLabel)

                $apiBox = New-Object System.Windows.Forms.TextBox
                $apiBox.Location = New-Object System.Drawing.Point(70, ($contentYPos - 3))
                $apiBox.Size = New-Object System.Drawing.Size(380, 25)
                $apiBox.Font = New-Object System.Drawing.Font($script:GuiConfig.FontFamily, 9)
                $apiBox.Name = "apikey_$serverKey"
                $apiBox.UseSystemPasswordChar = $true
                $configPanel.Controls.Add($apiBox)

                if ($server.ApiKeyUrl) {
                    $linkLabel = New-Object System.Windows.Forms.LinkLabel
                    $linkLabel.Text = "Get API Key"
                    $linkLabel.Location = New-Object System.Drawing.Point(455, $contentYPos)
                    $linkLabel.AutoSize = $true
                    $linkLabel.LinkColor = $script:GuiConfig.AccentColor
                    $linkLabel.Tag = $server.ApiKeyUrl
                    $linkLabel.Add_Click({
                        Start-Process $this.Tag
                    })
                    $configPanel.Controls.Add($linkLabel)
                }
            }

            $scrollPanel.Controls.Add($configPanel)
            $yPos += $panelHeight + 10
        }
    }

    if ($yPos -eq 15) {
        $noConfigLabel = New-StyledLabel -Text "No additional configuration required for selected servers." -FontSize 11
        $noConfigLabel.Location = New-Object System.Drawing.Point(15, 20)
        $scrollPanel.Controls.Add($noConfigLabel)
    }
}

function New-InstallationPage {
    $panel = New-Object System.Windows.Forms.Panel
    $panel.Dock = [System.Windows.Forms.DockStyle]::Fill
    $panel.BackColor = $script:GuiConfig.PrimaryColor

    # Title
    $title = New-StyledLabel -Text "Installation Progress" -FontSize 18 -FontStyle ([System.Drawing.FontStyle]::Bold)
    $title.Location = New-Object System.Drawing.Point(50, 30)
    $panel.Controls.Add($title)

    # Progress bar
    $progressBar = New-Object System.Windows.Forms.ProgressBar
    $progressBar.Location = New-Object System.Drawing.Point(50, 80)
    $progressBar.Size = New-Object System.Drawing.Size(580, 25)
    $progressBar.Style = [System.Windows.Forms.ProgressBarStyle]::Continuous
    $progressBar.Name = "ProgressBar"
    $panel.Controls.Add($progressBar)

    # Status label
    $statusLabel = New-StyledLabel -Text "Ready to install..." -FontSize 10
    $statusLabel.Location = New-Object System.Drawing.Point(50, 115)
    $statusLabel.Name = "StatusLabel"
    $panel.Controls.Add($statusLabel)

    # Log text box
    $logBox = New-Object System.Windows.Forms.TextBox
    $logBox.Location = New-Object System.Drawing.Point(50, 150)
    $logBox.Size = New-Object System.Drawing.Size(580, 280)
    $logBox.Multiline = $true
    $logBox.ScrollBars = [System.Windows.Forms.ScrollBars]::Vertical
    $logBox.ReadOnly = $true
    $logBox.BackColor = $script:GuiConfig.SecondaryColor
    $logBox.ForeColor = $script:GuiConfig.TextColor
    $logBox.Font = New-Object System.Drawing.Font("Consolas", 9)
    $logBox.Name = "LogBox"
    $panel.Controls.Add($logBox)

    return $panel
}

function New-CompletionPage {
    $panel = New-Object System.Windows.Forms.Panel
    $panel.Dock = [System.Windows.Forms.DockStyle]::Fill
    $panel.BackColor = $script:GuiConfig.PrimaryColor

    # Title
    $title = New-StyledLabel -Text "Installation Complete!" -FontSize 24 -FontStyle ([System.Drawing.FontStyle]::Bold) -ForeColor $script:GuiConfig.SuccessColor
    $title.Location = New-Object System.Drawing.Point(50, 50)
    $panel.Controls.Add($title)

    # Description
    $desc = New-Object System.Windows.Forms.Label
    $desc.Text = @"
The Claude MCP Installer has finished setting up your environment.

What was installed:
"@
    $desc.Font = New-Object System.Drawing.Font($script:GuiConfig.FontFamily, 11)
    $desc.ForeColor = $script:GuiConfig.TextColor
    $desc.BackColor = [System.Drawing.Color]::Transparent
    $desc.Location = New-Object System.Drawing.Point(50, 100)
    $desc.Size = New-Object System.Drawing.Size(580, 60)
    $panel.Controls.Add($desc)

    # Installed items list
    $installedList = New-Object System.Windows.Forms.ListBox
    $installedList.Location = New-Object System.Drawing.Point(50, 165)
    $installedList.Size = New-Object System.Drawing.Size(580, 120)
    $installedList.BackColor = $script:GuiConfig.SecondaryColor
    $installedList.ForeColor = $script:GuiConfig.SuccessColor
    $installedList.Font = New-Object System.Drawing.Font($script:GuiConfig.FontFamily, 10)
    $installedList.BorderStyle = [System.Windows.Forms.BorderStyle]::None
    $installedList.Name = "InstalledList"
    $panel.Controls.Add($installedList)

    # Next steps
    $nextSteps = New-Object System.Windows.Forms.Label
    $nextSteps.Text = @"
Next Steps:
    1. Restart Claude Desktop to load the new MCP servers
    2. Use 'claude' command in terminal to access Claude Code CLI
    3. Check the log file if you encounter any issues

Configuration saved to:
    %APPDATA%\Claude\claude_desktop_config.json
"@
    $nextSteps.Font = New-Object System.Drawing.Font($script:GuiConfig.FontFamily, 10)
    $nextSteps.ForeColor = $script:GuiConfig.TextColor
    $nextSteps.BackColor = [System.Drawing.Color]::Transparent
    $nextSteps.Location = New-Object System.Drawing.Point(50, 300)
    $nextSteps.Size = New-Object System.Drawing.Size(580, 130)
    $panel.Controls.Add($nextSteps)

    return $panel
}

# ============================================================================
# Main Form Creation
# ============================================================================

function Show-InstallerWizard {
    # Create main form
    $form = New-Object System.Windows.Forms.Form
    $form.Text = "Claude MCP Installer"
    $form.Size = New-Object System.Drawing.Size($script:GuiConfig.WindowWidth, $script:GuiConfig.WindowHeight)
    $form.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
    $form.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedSingle
    $form.MaximizeBox = $false
    $form.BackColor = $script:GuiConfig.PrimaryColor
    $form.Icon = [System.Drawing.SystemIcons]::Application

    # Create page container
    $pageContainer = New-Object System.Windows.Forms.Panel
    $pageContainer.Location = New-Object System.Drawing.Point(0, 0)
    $pageContainer.Size = New-Object System.Drawing.Size($script:GuiConfig.WindowWidth, ($script:GuiConfig.WindowHeight - 80))
    $pageContainer.Name = "PageContainer"
    $form.Controls.Add($pageContainer)

    # Create pages
    $pages = @(
        (New-WelcomePage),
        (New-PrerequisitesPage),
        (New-ServerSelectionPage),
        (New-ConfigurationPage),
        (New-InstallationPage),
        (New-CompletionPage)
    )

    foreach ($page in $pages) {
        $page.Visible = $false
        $pageContainer.Controls.Add($page)
    }

    # Navigation panel
    $navPanel = New-Object System.Windows.Forms.Panel
    $navPanel.Location = New-Object System.Drawing.Point(0, ($script:GuiConfig.WindowHeight - 85))
    $navPanel.Size = New-Object System.Drawing.Size($script:GuiConfig.WindowWidth, 50)
    $navPanel.BackColor = $script:GuiConfig.SecondaryColor
    $form.Controls.Add($navPanel)

    # Navigation buttons
    $backBtn = New-StyledButton -Text "< Back" -Width 100 -BackColor $script:GuiConfig.SecondaryColor
    $backBtn.Location = New-Object System.Drawing.Point(20, 8)
    $backBtn.Name = "BackBtn"
    $navPanel.Controls.Add($backBtn)

    $nextBtn = New-StyledButton -Text "Next >" -Width 100
    $nextBtn.Location = New-Object System.Drawing.Point(460, 8)
    $nextBtn.Name = "NextBtn"
    $navPanel.Controls.Add($nextBtn)

    $cancelBtn = New-StyledButton -Text "Cancel" -Width 100 -BackColor ([System.Drawing.Color]::FromArgb(100, 100, 100))
    $cancelBtn.Location = New-Object System.Drawing.Point(570, 8)
    $cancelBtn.Name = "CancelBtn"
    $navPanel.Controls.Add($cancelBtn)

    # Page navigation function
    $showPage = {
        param($PageIndex)

        for ($i = 0; $i -lt $pages.Count; $i++) {
            $pages[$i].Visible = ($i -eq $PageIndex)
        }

        $script:CurrentPage = $PageIndex

        # Update navigation buttons
        $backBtn.Enabled = ($PageIndex -gt 0 -and $PageIndex -lt 5)

        switch ($PageIndex) {
            0 { $nextBtn.Text = "Next >"; $nextBtn.Enabled = $true }
            1 {
                $nextBtn.Text = "Next >"
                $nextBtn.Enabled = $true
                Update-PrerequisitesStatus -Panel $pages[1]
            }
            2 { $nextBtn.Text = "Next >"; $nextBtn.Enabled = $true }
            3 {
                $nextBtn.Text = "Install"
                $nextBtn.Enabled = $true
                # Get selected servers
                $scrollPanel = $pages[2].Controls["ServerScrollPanel"]
                $script:SelectedServers = @()
                foreach ($control in $scrollPanel.Controls) {
                    foreach ($checkbox in $control.Controls) {
                        if ($checkbox -is [System.Windows.Forms.CheckBox] -and $checkbox.Checked) {
                            $script:SelectedServers += $checkbox.Tag
                        }
                    }
                }
                Update-ConfigurationPage -Panel $pages[3] -SelectedServers $script:SelectedServers
            }
            4 {
                $nextBtn.Text = "Next >"
                $nextBtn.Enabled = $false
                $backBtn.Enabled = $false
            }
            5 {
                $nextBtn.Text = "Finish"
                $nextBtn.Enabled = $true
                $backBtn.Enabled = $false
            }
        }
    }

    # Button click handlers
    $backBtn.Add_Click({
        if ($script:CurrentPage -gt 0) {
            & $showPage ($script:CurrentPage - 1)
        }
    })

    $nextBtn.Add_Click({
        switch ($script:CurrentPage) {
            3 {
                # Start installation
                & $showPage 4
                Start-Installation -Pages $pages -Form $form -ShowPage $showPage
            }
            5 {
                $form.Close()
            }
            default {
                if ($script:CurrentPage -lt ($pages.Count - 1)) {
                    & $showPage ($script:CurrentPage + 1)
                }
            }
        }
    })

    $cancelBtn.Add_Click({
        $result = [System.Windows.Forms.MessageBox]::Show(
            "Are you sure you want to cancel the installation?",
            "Cancel Installation",
            [System.Windows.Forms.MessageBoxButtons]::YesNo,
            [System.Windows.Forms.MessageBoxIcon]::Question
        )
        if ($result -eq [System.Windows.Forms.DialogResult]::Yes) {
            $form.Close()
        }
    })

    # Prerequisites page handlers
    $prereqPage = $pages[1]
    $refreshBtn = $prereqPage.Controls["RefreshBtn"]
    $refreshBtn.Add_Click({
        Update-PrerequisitesStatus -Panel $pages[1]
    })

    $installPrereqBtn = $prereqPage.Controls["InstallPrereqBtn"]
    $installPrereqBtn.Add_Click({
        $this.Enabled = $false
        $this.Text = "Installing..."

        # Install missing prerequisites
        if (-not (Test-NodeJS)) { $null = Install-NodeJS }
        if (-not (Test-Python)) { $null = Install-Python }
        if (-not (Test-UV)) { $null = Install-UV }
        if (-not (Test-ClaudeCode)) { $null = Install-ClaudeCode }

        Update-PrerequisitesStatus -Panel $pages[1]
        $this.Text = "Install Missing"
        $this.Enabled = $true
    })

    # Server selection page handlers
    $serverPage = $pages[2]
    $selectAllBtn = $serverPage.Controls["SelectAllBtn"]
    $selectAllBtn.Add_Click({
        $scrollPanel = $pages[2].Controls["ServerScrollPanel"]
        foreach ($control in $scrollPanel.Controls) {
            foreach ($checkbox in $control.Controls) {
                if ($checkbox -is [System.Windows.Forms.CheckBox]) {
                    $checkbox.Checked = $true
                }
            }
        }
    })

    $deselectAllBtn = $serverPage.Controls["DeselectAllBtn"]
    $deselectAllBtn.Add_Click({
        $scrollPanel = $pages[2].Controls["ServerScrollPanel"]
        foreach ($control in $scrollPanel.Controls) {
            foreach ($checkbox in $control.Controls) {
                if ($checkbox -is [System.Windows.Forms.CheckBox]) {
                    $checkbox.Checked = $false
                }
            }
        }
    })

    # Show first page
    & $showPage 0

    # Show form
    $form.ShowDialog() | Out-Null
}

function Start-Installation {
    param($Pages, $Form, $ShowPage)

    $installPage = $Pages[4]
    $progressBar = $installPage.Controls["ProgressBar"]
    $statusLabel = $installPage.Controls["StatusLabel"]
    $logBox = $installPage.Controls["LogBox"]

    $completionPage = $Pages[5]
    $installedList = $completionPage.Controls["InstalledList"]

    # Helper to update UI
    $updateUI = {
        param($Status, $Progress, $LogMessage)
        $statusLabel.Text = $Status
        $progressBar.Value = [Math]::Min($Progress, 100)
        if ($LogMessage) {
            $logBox.AppendText("$LogMessage`r`n")
        }
        $Form.Refresh()
        [System.Windows.Forms.Application]::DoEvents()
    }

    try {
        & $updateUI "Initializing..." 5 "[INFO] Starting installation..."

        $totalServers = $script:SelectedServers.Count
        if ($totalServers -eq 0) {
            & $updateUI "No servers selected" 100 "[WARN] No MCP servers were selected for installation."
            Start-Sleep -Seconds 2
            & $ShowPage 5
            return
        }

        $mcpConfig = @{
            mcpServers = @{}
        }

        $currentServer = 0
        foreach ($serverKey in $script:SelectedServers) {
            $currentServer++
            $progress = [int](($currentServer / $totalServers) * 80) + 10
            $server = $MCPServers[$serverKey]

            & $updateUI "Configuring $($server.Name)..." $progress "[INFO] Processing: $($server.Name)"

            if ($null -eq $server.Config) {
                & $updateUI "Configuring $($server.Name)..." $progress "[WARN] Skipping $($server.Name) - requires manual configuration"
                continue
            }

            $config = @{}

            # Handle embedded watchdog script
            if ($server.Type -eq "embedded-python") {
                & $updateUI "Installing Watchdog script..." $progress "[INFO] Installing Conversation Watchdog script..."
                $watchdogPath = Install-WatchdogScript

                if ($server.PythonPackages) {
                    & $updateUI "Installing Python packages..." $progress "[INFO] Installing Python packages: $($server.PythonPackages -join ', ')"
                    $null = Install-PythonPackages -Packages $server.PythonPackages
                }

                $config.command = $server.Config.command
                $config.args = @($watchdogPath)
            }
            elseif ($server.RequiresPath) {
                # Get path from config page
                $configPage = $Pages[3]
                $scrollPanel = $configPage.Controls["ConfigScrollPanel"]
                $pathBox = $null
                foreach ($ctrl in $scrollPanel.Controls) {
                    $pathBox = $ctrl.Controls["path_$serverKey"]
                    if ($pathBox) { break }
                }

                $path = if ($pathBox) { $pathBox.Text } else { $server.DefaultPath }

                $config.command = $server.Config.command -replace '\{PATH\}', $path
                $config.args = $server.Config.args | ForEach-Object { $_ -replace '\{PATH\}', $path }
            }
            else {
                $config.command = $server.Config.command
                $config.args = $server.Config.args
            }

            # Handle API key
            if ($server.RequiresApiKey) {
                $configPage = $Pages[3]
                $scrollPanel = $configPage.Controls["ConfigScrollPanel"]
                $apiBox = $null
                foreach ($ctrl in $scrollPanel.Controls) {
                    $apiBox = $ctrl.Controls["apikey_$serverKey"]
                    if ($apiBox) { break }
                }

                $apiKey = if ($apiBox) { $apiBox.Text } else { "" }

                if ([string]::IsNullOrWhiteSpace($apiKey)) {
                    & $updateUI "Configuring $($server.Name)..." $progress "[WARN] Skipping $($server.Name) - no API key provided"
                    continue
                }

                if ($server.Config.env) {
                    $config.env = @{}
                    foreach ($envKey in $server.Config.env.Keys) {
                        $config.env[$envKey] = $server.Config.env[$envKey] -replace '\{API_KEY\}', $apiKey
                    }
                }
            }
            elseif ($server.Config.env) {
                $config.env = $server.Config.env
            }

            $mcpConfig.mcpServers[$serverKey] = $config
            $installedList.Items.Add($server.Name)
            & $updateUI "Configured $($server.Name)" $progress "[OK] Configured: $($server.Name)"
        }

        # Save configuration
        & $updateUI "Saving configuration..." 90 "[INFO] Saving configuration to claude_desktop_config.json..."

        if ($mcpConfig.mcpServers.Count -gt 0) {
            $configPath = Save-MCPConfig -Config $mcpConfig
            & $updateUI "Configuration saved!" 100 "[OK] Configuration saved to: $configPath"
        }
        else {
            & $updateUI "No servers configured" 100 "[WARN] No MCP servers were configured"
        }

        & $updateUI "Installation complete!" 100 "[INFO] Installation completed successfully!"
        Start-Sleep -Seconds 1

        # Show completion page
        & $ShowPage 5
    }
    catch {
        & $updateUI "Error: $($_.Exception.Message)" $progressBar.Value "[ERROR] $($_.Exception.Message)"
        [System.Windows.Forms.MessageBox]::Show(
            "An error occurred during installation:`n`n$($_.Exception.Message)",
            "Installation Error",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        )
    }
}

# ============================================================================
# Entry Point
# ============================================================================

# Check if we're being dot-sourced or run directly
if ($MyInvocation.InvocationName -ne '.') {
    Show-InstallerWizard
}
