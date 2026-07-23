param([Parameter(Mandatory = $true)][Int64]$Hwnd)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
$element = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]$Hwnd)
if ($null -eq $element) {
    @{ found = $false } | ConvertTo-Json -Compress
    exit 2
}
$children = $element.FindAll(
    [System.Windows.Automation.TreeScope]::Descendants,
    [System.Windows.Automation.Condition]::TrueCondition
)
$items = @()
foreach ($child in $children) {
    try {
        if ($child.Current.Name) {
            $items += @{
                name = $child.Current.Name
                control_type = $child.Current.ControlType.ProgrammaticName
            }
        }
    } catch { }
}
@{
    found = $true
    root_name = $element.Current.Name
    root_control_type = $element.Current.ControlType.ProgrammaticName
    descendants = $items
} | ConvertTo-Json -Depth 5 -Compress

