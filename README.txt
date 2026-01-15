============================================================================
Claude Code & MCP Servers Installer
============================================================================
Version: 2.0.0
Author: Quality By Design NV

DESCRIPTION
-----------
This installer sets up Claude Code CLI and configures MCP (Model Context
Protocol) servers for Claude Desktop. It handles all prerequisites and
allows users to choose which MCP servers to install.

REQUIREMENTS
------------
- Windows 10/11
- Internet connection
- Administrator rights (recommended, but not required for all features)

AVAILABLE MCP SERVERS
---------------------
1. PowerBI Modeling MCP
   - Manipulate Power BI semantic models via Tabular Editor
   - Requires: VS Code PowerBI extension installed

2. Microsoft 365 MCP
   - Access Outlook, OneDrive, Calendar, Teams, and other M365 services
   - Requires: Node.js

3. Teams MCP
   - Microsoft Teams integration for chats and channels
   - Requires: Node.js

4. Excel MCP
   - Read and write Excel files, create tables and charts
   - Requires: Node.js

5. Word Document MCP
   - Create and edit Word documents
   - Requires: Python, UV

6. PowerPoint MCP
   - Create and edit PowerPoint presentations
   - Requires: Python, UV

7. TeamTailor MCP
   - TeamTailor recruitment platform integration
   - Requires: Node.js, TeamTailor API key

8. Conversation Watchdog MCP
   - Monitors conversations for truncation and enables recovery
   - Requires: Python, custom script

9. HubSpot MCP
   - HubSpot CRM integration (requires separate marketplace installation)

HOW TO USE
----------
Method 1: GUI Wizard (Recommended)
  - Double-click "Install-ClaudeMCP-GUI.bat"
  - Follow the step-by-step wizard interface
  - Modern visual interface with progress tracking

Method 2: Command-line interface
  - Double-click "Install-ClaudeMCP.bat"
  - Follow the on-screen instructions in terminal

Method 3: Run PowerShell directly
  - Open PowerShell
  - Navigate to this folder
  - Run: .\Install-ClaudeMCP.ps1

Method 4: With parameters
  - Skip prerequisites check: .\Install-ClaudeMCP.ps1 -SkipPrerequisites
  - GUI version: .\Install-ClaudeMCP-GUI.ps1

WHAT THE INSTALLER DOES
-----------------------
1. Checks and installs Node.js (if not present)
2. Checks and installs Python (if not present)
3. Checks and installs UV package manager (if not present)
4. Installs Claude Code CLI via npm
5. Presents a menu to select which MCP servers to install
6. Prompts for paths and API keys as needed
7. Creates/updates the claude_desktop_config.json file
8. Backs up any existing configuration

CONFIGURATION FILE LOCATION
---------------------------
Windows: %APPDATA%\Claude\claude_desktop_config.json

TROUBLESHOOTING
---------------
Issue: "Script cannot be loaded because running scripts is disabled"
Solution: Run the .bat file instead, or execute:
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

Issue: Node.js installation fails
Solution: Install manually from https://nodejs.org

Issue: Python installation fails
Solution: Install manually from https://python.org

Issue: MCP server not working after installation
Solution:
  1. Restart Claude Desktop completely
  2. Check the config file for syntax errors
  3. Verify all paths are correct

MANUAL CONFIGURATION
--------------------
If you need to manually edit the MCP configuration, the file is located at:
  %APPDATA%\Claude\claude_desktop_config.json

Example configuration entry:
{
  "mcpServers": {
    "excel": {
      "command": "npx",
      "args": ["--yes", "@negokaz/excel-mcp-server"],
      "env": {
        "EXCEL_MCP_PAGING_CELLS_LIMIT": "4000"
      }
    }
  }
}

SUPPORT
-------
For issues with this installer, contact Quality By Design NV.
For Claude Code issues, visit: https://github.com/anthropics/claude-code/issues

============================================================================
