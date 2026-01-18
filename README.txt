============================================================================
Claude Code & MCP Servers Installer
============================================================================
Version: 2.2.0
Author: Ackie

DESCRIPTION
-----------
This installer sets up Claude Code CLI and configures MCP (Model Context
Protocol) servers for Claude Desktop. It handles all prerequisites and
allows users to choose which MCP servers to install.

NEW IN VERSION 2.2: Added automatic installation of VS Code and PowerBI
extension prerequisites for PowerBI Modeling MCP. The installer now
automatically installs VS Code and the PowerBI Modeling MCP extension
when selected.

NEW IN VERSION 2.1: Added Notion and Airtable MCP servers, improved GUI
configuration panel layout, and fixed encoding issues.

QUICK INSTALL FROM GITHUB
-------------------------
Run this one-liner in PowerShell (from a folder like Desktop or Documents):

  cd $HOME\Desktop; if (Test-Path Claude-MCP-Installer) { Remove-Item -Recurse -Force Claude-MCP-Installer }; git clone https://github.com/Ackie1980/Claude-MCP-Installer.git; cd Claude-MCP-Installer; powershell -ExecutionPolicy Bypass -File .\Install-ClaudeMCP-GUI.ps1

Or for the command-line version:

  cd $HOME\Desktop; if (Test-Path Claude-MCP-Installer) { Remove-Item -Recurse -Force Claude-MCP-Installer }; git clone https://github.com/Ackie1980/Claude-MCP-Installer.git; cd Claude-MCP-Installer; powershell -ExecutionPolicy Bypass -File .\Install-ClaudeMCP.ps1

NOTE: The command changes to your Desktop folder first, removes any existing
      Claude-MCP-Installer folder, then clones fresh. You can change Desktop
      to any folder you prefer (e.g., Documents).

REQUIREMENTS
------------
- Windows 10/11
- Internet connection
- Administrator rights (recommended, but not required for all features)

AVAILABLE MCP SERVERS
---------------------
1. PowerBI Modeling MCP
   - Manipulate Power BI semantic models via Tabular Editor
   - Requires: VS Code, VS Code PowerBI extension (auto-installed)

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

7. Notion MCP
   - Official Notion MCP Server - Access and manage Notion workspaces,
     pages, databases and blocks
   - Requires: Node.js, Notion Integration Token

8. Airtable MCP
   - Read and write to Airtable bases, tables and records
   - Requires: Node.js, Airtable Personal Access Token

9. TeamTailor MCP
   - TeamTailor recruitment platform integration
   - Requires: Node.js, TeamTailor API key

10. Google Cloud Storage MCP
    - Interact with Google Cloud Storage buckets
    - Requires: Node.js, Google Cloud Project ID

11. Conversation Watchdog MCP
    - Monitors conversations for truncation and enables recovery
    - Requires: Python, custom script

12. HubSpot MCP
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
  - Run: powershell -ExecutionPolicy Bypass -File .\Install-ClaudeMCP.ps1

Method 4: With parameters
  - Skip prerequisites check: .\Install-ClaudeMCP.ps1 -SkipPrerequisites
  - GUI version: .\Install-ClaudeMCP-GUI.ps1

WHAT THE INSTALLER DOES
-----------------------
1. Checks and installs Node.js (if not present)
2. Checks and installs Python (if not present)
3. Checks and installs UV package manager (if not present)
4. Installs Claude Code CLI via npm
5. Checks and installs VS Code (if not present)
6. Checks and installs VS Code PowerBI extension (if not present)
7. Presents a menu to select which MCP servers to install
8. Prompts for paths and API keys as needed
9. Creates/updates the claude_desktop_config.json file
10. Backs up any existing configuration
11. PRESERVES existing MCP configurations (won't remove your current MCPs)

CONFIGURATION FILE LOCATION
---------------------------
Windows: %APPDATA%\Claude\claude_desktop_config.json

TROUBLESHOOTING
---------------
Issue: "destination path 'Claude-MCP-Installer' already exists" when running
       git clone
Solution: The folder already exists from a previous install attempt. Either:
  1. Use the updated install command above (includes automatic cleanup), or
  2. Manually delete the folder first:
       Remove-Item -Recurse -Force Claude-MCP-Installer
  3. Or navigate into the existing folder and pull updates:
       cd Claude-MCP-Installer; git pull

Issue: "cd : Cannot find path" or running from C:\Windows\system32
Solution: Don't run the install command from system32. First change to a user
  folder like Desktop:
    cd $HOME\Desktop
  Then run the install command.

Issue: "Script cannot be loaded because running scripts is disabled"
Solution: Use the -ExecutionPolicy Bypass flag:
  powershell -ExecutionPolicy Bypass -File .\Install-ClaudeMCP-GUI.ps1

Issue: Node.js installation fails
Solution: Install manually from https://nodejs.org

Issue: Python installation fails
Solution: Install manually from https://python.org

Issue: MCP server not working after installation
Solution:
  1. Restart Claude Desktop completely
  2. Check the config file for syntax errors
  3. Verify all paths are correct

Issue: "Test-NodeJS is not recognized" error in GUI
Solution: Make sure both Install-ClaudeMCP.ps1 and Install-ClaudeMCP-GUI.ps1
are in the same directory

MANUAL CONFIGURATION
--------------------
If you need to manually edit the MCP configuration, the file is located at:
  %APPDATA%\Claude\claude_desktop_config.json

Example configuration entries:

{
  "mcpServers": {
    "excel": {
      "command": "npx",
      "args": ["--yes", "@negokaz/excel-mcp-server"],
      "env": {
        "EXCEL_MCP_PAGING_CELLS_LIMIT": "4000"
      }
    },
    "notion": {
      "command": "npx",
      "args": ["-y", "@notionhq/notion-mcp-server"],
      "env": {
        "NOTION_TOKEN": "ntn_your_token_here"
      }
    },
    "airtable": {
      "command": "npx",
      "args": ["-y", "airtable-mcp-server"],
      "env": {
        "AIRTABLE_API_KEY": "your_api_key_here"
      }
    }
  }
}

API KEY SETUP LINKS
-------------------
- Notion: https://www.notion.so/profile/integrations
- Airtable: https://airtable.com/create/tokens
- TeamTailor: Contact your TeamTailor administrator
- Google Cloud: https://console.cloud.google.com

SUPPORT
-------
For issues with this installer: https://github.com/Ackie1980/Claude-MCP-Installer/issues
For Claude Code issues: https://github.com/anthropics/claude-code/issues
GitHub Repository: https://github.com/Ackie1980/Claude-MCP-Installer

============================================================================
