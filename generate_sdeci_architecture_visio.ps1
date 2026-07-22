param(
    [string]$OutputPath = "",
    [switch]$OpenVisio,
    [switch]$FixedName
)

$ErrorActionPreference = "Stop"

function Join-Lines {
    param([string[]]$Lines)
    return [string]::Join([Environment]::NewLine, $Lines)
}

function Style-Text {
    param(
        [object]$Shape,
        [double]$Size = 8,
        [string]$Color = "RGB(30,41,59)",
        [int]$HAlign = 1,
        [int]$VAlign = 1
    )
    $Shape.CellsU("Char.Size").FormulaU = "$Size pt"
    $Shape.CellsU("Char.Color").FormulaU = $Color
    $Shape.CellsU("Para.HorzAlign").FormulaU = "$HAlign"
    $Shape.CellsU("VerticalAlign").FormulaU = "$VAlign"
    $Shape.CellsU("TxtWidth").FormulaU = "Width*0.92"
}

function Box {
    param(
        [object]$Page,
        [double]$X,
        [double]$Y,
        [double]$W,
        [double]$H,
        [string]$Text = "",
        [string]$Fill = "RGB(255,255,255)",
        [string]$Line = "RGB(30,41,59)",
        [double]$FontSize = 8,
        [double]$Weight = 1.1,
        [double]$Round = 0.08,
        [bool]$Dashed = $false
    )
    $s = $Page.DrawRectangle($X, $Y, $X + $W, $Y + $H)
    $s.Text = $Text
    $s.CellsU("FillForegnd").FormulaU = $Fill
    $s.CellsU("LineColor").FormulaU = $Line
    $s.CellsU("LineWeight").FormulaU = "$Weight pt"
    $s.CellsU("Rounding").FormulaU = "$Round in"
    if ($Dashed) {
        $s.CellsU("LinePattern").FormulaU = "2"
    }
    Style-Text -Shape $s -Size $FontSize
    return $s
}

function Text {
    param(
        [object]$Page,
        [double]$X,
        [double]$Y,
        [double]$W,
        [double]$H,
        [string]$Text,
        [double]$FontSize = 8,
        [string]$Color = "RGB(30,41,59)",
        [int]$HAlign = 1,
        [int]$VAlign = 1
    )
    $s = $Page.DrawRectangle($X, $Y, $X + $W, $Y + $H)
    $s.Text = $Text
    $s.CellsU("FillPattern").FormulaU = "0"
    $s.CellsU("LinePattern").FormulaU = "0"
    Style-Text -Shape $s -Size $FontSize -Color $Color -HAlign $HAlign -VAlign $VAlign
    return $s
}

function Arrow {
    param(
        [object]$Page,
        [double]$X1,
        [double]$Y1,
        [double]$X2,
        [double]$Y2,
        [string]$Color = "RGB(30,41,59)",
        [double]$Weight = 1.6,
        [bool]$Dashed = $false,
        [string]$Label = ""
    )
    $l = $Page.DrawLine($X1, $Y1, $X2, $Y2)
    $l.CellsU("LineColor").FormulaU = $Color
    $l.CellsU("LineWeight").FormulaU = "$Weight pt"
    $l.CellsU("EndArrow").FormulaU = "13"
    if ($Dashed) {
        $l.CellsU("LinePattern").FormulaU = "2"
    }
    if ($Label -ne "") {
        Text -Page $Page -X (($X1+$X2)/2-0.38) -Y (($Y1+$Y2)/2+0.08) -W 0.76 -H 0.20 -Text $Label -FontSize 6.4 -Color $Color | Out-Null
    }
    return $l
}

function MatrixIcon {
    param(
        [object]$Page,
        [double]$X,
        [double]$Y,
        [int]$Rows = 5,
        [int]$Cols = 5,
        [double]$Cell = 0.12,
        [string[]]$Palette = @("RGB(47,92,158)", "RGB(242,167,111)", "RGB(247,210,168)", "RGB(64,132,88)")
    )
    for ($r = 0; $r -lt $Rows; $r++) {
        for ($c = 0; $c -lt $Cols; $c++) {
            $color = $Palette[($r * 3 + $c * 5) % $Palette.Count]
            Box -Page $Page -X ($X + $c*$Cell) -Y ($Y + ($Rows-1-$r)*$Cell) -W ($Cell*0.95) -H ($Cell*0.95) -Fill $color -Line "RGB(255,255,255)" -Weight 0.2 -Round 0.00 | Out-Null
        }
    }
}

function VectorIcon {
    param(
        [object]$Page,
        [double]$X,
        [double]$Y,
        [int]$N = 7,
        [string]$Color = "RGB(53,95,165)",
        [double]$W = 0.18,
        [double]$H = 0.16
    )
    for ($i = 0; $i -lt $N; $i++) {
        Box -Page $Page -X $X -Y ($Y + $i*($H+0.025)) -W $W -H $H -Fill $Color -Line "RGB(255,255,255)" -Weight 0.25 -Round 0.01 | Out-Null
    }
}

function NodeGraphIcon {
    param(
        [object]$Page,
        [double]$X,
        [double]$Y,
        [string]$Color = "RGB(53,95,165)"
    )
    $pts = @(
        [pscustomobject]@{ X = $X + 0.15; Y = $Y + 0.15 },
        [pscustomobject]@{ X = $X + 0.55; Y = $Y + 0.28 },
        [pscustomobject]@{ X = $X + 0.88; Y = $Y + 0.10 },
        [pscustomobject]@{ X = $X + 0.30; Y = $Y + 0.62 },
        [pscustomobject]@{ X = $X + 0.78; Y = $Y + 0.70 },
        [pscustomobject]@{ X = $X + 0.58; Y = $Y + 0.98 }
    )
    $edges = @(@(0,1),@(1,2),@(0,3),@(3,4),@(4,2),@(1,4),@(4,5),@(3,5))
    foreach ($e in $edges) {
        $a = $pts[$e[0]]
        $b = $pts[$e[1]]
        Arrow -Page $Page -X1 $a.X -Y1 $a.Y -X2 $b.X -Y2 $b.Y -Color "RGB(137,148,162)" -Weight 0.7 | Out-Null
    }
    foreach ($p in $pts) {
        $o = $Page.DrawOval($p.X-0.045, $p.Y-0.045, $p.X+0.045, $p.Y+0.045)
        $o.CellsU("FillForegnd").FormulaU = $Color
        $o.CellsU("LineColor").FormulaU = "RGB(255,255,255)"
        $o.CellsU("LineWeight").FormulaU = "0.4 pt"
    }
}

function BrainIcon {
    param([object]$Page, [double]$X, [double]$Y, [string]$Fill = "RGB(232,236,241)")
    $b1 = $Page.DrawOval($X, $Y+0.05, $X+0.55, $Y+0.80)
    $b2 = $Page.DrawOval($X+0.38, $Y+0.05, $X+0.93, $Y+0.80)
    foreach ($b in @($b1,$b2)) {
        $b.CellsU("FillForegnd").FormulaU = $Fill
        $b.CellsU("LineColor").FormulaU = "RGB(75,85,99)"
        $b.CellsU("LineWeight").FormulaU = "0.8 pt"
    }
    $mid = $Page.DrawLine($X+0.47, $Y+0.12, $X+0.47, $Y+0.78)
    $mid.CellsU("LineColor").FormulaU = "RGB(110,123,140)"
    $mid.CellsU("LinePattern").FormulaU = "2"
    $mid.CellsU("LineWeight").FormulaU = "0.6 pt"
}

function PlusIcon {
    param([object]$Page, [double]$X, [double]$Y)
    $c = $Page.DrawOval($X, $Y, $X+0.22, $Y+0.22)
    $c.CellsU("FillForegnd").FormulaU = "RGB(255,255,255)"
    $c.CellsU("LineColor").FormulaU = "RGB(30,41,59)"
    $c.CellsU("LineWeight").FormulaU = "1 pt"
    Text -Page $Page -X $X -Y $Y -W 0.22 -H 0.22 -Text "+" -FontSize 12 -Color "RGB(30,41,59)" | Out-Null
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $outDir = Join-Path $root "outputs"
    if (-not (Test-Path $outDir)) {
        New-Item -ItemType Directory -Path $outDir | Out-Null
    }
    if ($FixedName) {
        $OutputPath = Join-Path $outDir "S-DeCI_default_architecture.vsdx"
    }
    else {
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $OutputPath = Join-Path $outDir "S-DeCI_main_figure_$stamp.vsdx"
    }
}

try {
    $visio = New-Object -ComObject Visio.Application
}
catch {
    throw "Microsoft Visio COM automation is not available. Please run this script on a machine with Microsoft Visio installed. Original error: $($_.Exception.Message)"
}

$visio.Visible = [bool]$OpenVisio
$doc = $visio.Documents.Add("")
$page = $visio.ActivePage
$page.Name = "S-DeCI main figure"
$page.PageSheet.CellsU("PageWidth").FormulaU = "17.8 in"
$page.PageSheet.CellsU("PageHeight").FormulaU = "9.6 in"

$ink = "RGB(30,41,59)"
$dash = "RGB(42,62,90)"
$orange = "RGB(237,125,49)"
$green = "RGB(106,168,79)"
$blue = "RGB(61,133,198)"
$red = "RGB(204,0,0)"
$purple = "RGB(126,103,173)"
$darkGreen = "RGB(50,92,38)"
$softBlue = "RGB(232,241,252)"
$softGreen = "RGB(231,245,235)"
$softYellow = "RGB(255,246,221)"
$softPurple = "RGB(239,235,252)"
$softRed = "RGB(255,235,235)"

Text -Page $page -X 0.30 -Y 9.15 -W 17.2 -H 0.28 -Text "S-DeCI: Structure-aware Dynamic Causal Inference for fMRI Classification" -FontSize 15 -Color $ink | Out-Null

# Legend.
Box -Page $page -X 13.05 -Y 8.42 -W 4.20 -H 0.62 -Text "" -Fill "RGB(255,255,255)" -Line $dash -Weight 1.1 -Dashed $true -Round 0.08 | Out-Null
Arrow -Page $page -X1 13.25 -Y1 8.78 -X2 13.95 -Y2 8.78 -Color $orange -Weight 1.5 | Out-Null
Text -Page $page -X 14.05 -Y 8.68 -W 1.25 -H 0.20 -Text "main flow" -FontSize 8 -Color $ink -HAlign 0 | Out-Null
Arrow -Page $page -X1 13.25 -Y1 8.52 -X2 13.95 -Y2 8.52 -Color $green -Weight 1.3 | Out-Null
Text -Page $page -X 14.05 -Y 8.42 -W 1.50 -H 0.20 -Text "auxiliary/skip" -FontSize 8 -Color $ink -HAlign 0 | Out-Null
PlusIcon -Page $page -X 15.65 -Y 8.49
Text -Page $page -X 15.95 -Y 8.42 -W 0.95 -H 0.25 -Text "fusion" -FontSize 8 -Color $ink -HAlign 0 | Out-Null

# Panel A: data preparation.
Box -Page $page -X 0.35 -Y 7.25 -W 12.35 -H 1.72 -Text "" -Fill "RGB(255,255,255)" -Line $dash -Weight 1.4 -Dashed $true -Round 0.12 | Out-Null
Text -Page $page -X 9.15 -Y 8.48 -W 2.80 -H 0.28 -Text "(A) Data preparation" -FontSize 13 -Color $ink | Out-Null
BrainIcon -Page $page -X 0.75 -Y 7.65
Text -Page $page -X 0.62 -Y 7.33 -W 1.10 -H 0.18 -Text "fMRI" -FontSize 8 -Color $ink | Out-Null
Arrow -Page $page -X1 1.72 -Y1 8.05 -X2 2.40 -Y2 8.05 -Color $ink -Weight 1.1 | Out-Null
BrainIcon -Page $page -X 2.55 -Y 7.65 -Fill "RGB(220,237,255)"
Text -Page $page -X 2.43 -Y 7.33 -W 1.18 -H 0.18 -Text "AAL116" -FontSize 8 -Color $ink | Out-Null
Arrow -Page $page -X1 3.55 -Y1 8.05 -X2 4.20 -Y2 8.05 -Color $ink -Weight 1.1 | Out-Null
Box -Page $page -X 4.30 -Y 7.55 -W 1.75 -H 0.95 -Text "Time series X" -Fill $softBlue -Line "RGB(95,125,160)" -FontSize 8.5 -Round 0.03 | Out-Null
for ($i=0; $i -lt 5; $i++) {
    $x1 = 4.42
    $y1 = 7.78 + $i*0.10
    $line = $page.DrawLine($x1, $y1, 5.82, $y1 + 0.10*[Math]::Sin($i+1))
    $line.CellsU("LineColor").FormulaU = "RGB(90,130,110)"
    $line.CellsU("LineWeight").FormulaU = "0.5 pt"
}
Text -Page $page -X 4.52 -Y 8.50 -W 1.10 -H 0.20 -Text "seq_len=T" -FontSize 7 -Color "RGB(90,98,110)" | Out-Null
Arrow -Page $page -X1 6.10 -Y1 8.05 -X2 6.72 -Y2 8.05 -Color $ink -Weight 1.1 -Label "rFFT" | Out-Null
MatrixIcon -Page $page -X 6.90 -Y 7.65 -Rows 5 -Cols 7 -Cell 0.115
Text -Page $page -X 6.72 -Y 7.35 -W 1.35 -H 0.18 -Text "ALFF/fALFF" -FontSize 8 -Color $ink | Out-Null
Arrow -Page $page -X1 7.88 -Y1 8.05 -X2 8.55 -Y2 8.05 -Color $orange -Weight 1.4 | Out-Null
MatrixIcon -Page $page -X 8.78 -Y 7.70 -Rows 4 -Cols 8 -Cell 0.11
Text -Page $page -X 8.72 -Y 7.35 -W 1.60 -H 0.18 -Text "ROI feature C" -FontSize 8 -Color $ink | Out-Null
Arrow -Page $page -X1 10.00 -Y1 8.05 -X2 10.70 -Y2 8.05 -Color $orange -Weight 1.4 | Out-Null
MatrixIcon -Page $page -X 10.92 -Y 7.60 -Rows 6 -Cols 6 -Cell 0.105 -Palette @("RGB(251,191,36)","RGB(34,197,94)","RGB(59,130,246)","RGB(248,113,113)")
Text -Page $page -X 10.72 -Y 7.35 -W 1.78 -H 0.18 -Text "sample FC R" -FontSize 8 -Color $ink | Out-Null

# Panel B: causal graph construction.
Box -Page $page -X 0.35 -Y 3.75 -W 6.45 -H 3.22 -Text "" -Fill "RGB(255,255,255)" -Line $dash -Weight 1.4 -Dashed $true -Round 0.12 | Out-Null
Text -Page $page -X 4.60 -Y 6.58 -W 1.85 -H 0.28 -Text "(B) Causal graph" -FontSize 13 -Color $ink | Out-Null
Box -Page $page -X 0.75 -Y 6.10 -W 1.55 -H 0.42 -Text "Lag window" -Fill $softBlue -Line "RGB(100,116,139)" -FontSize 8.5 | Out-Null
for ($i=0; $i -lt 4; $i++) {
    Box -Page $page -X (0.82 + $i*0.25) -Y (5.55 + $i*0.08) -W 1.18 -H 0.35 -Text "" -Fill "RGB(231,240,255)" -Line "RGB(148,163,184)" -FontSize 1 -Round 0.02 | Out-Null
}
Text -Page $page -X 0.90 -Y 5.25 -W 1.40 -H 0.22 -Text "x(t-1...t-L)" -FontSize 8 -Color $ink | Out-Null
Arrow -Page $page -X1 2.35 -Y1 5.62 -X2 3.05 -Y2 5.62 -Color $orange -Weight 1.4 -Label "predict" | Out-Null
Box -Page $page -X 3.15 -Y 5.24 -W 1.20 -H 0.72 -Text "x(t)" -Fill $softGreen -Line "RGB(70,130,90)" -FontSize 9 | Out-Null
Arrow -Page $page -X1 4.40 -Y1 5.62 -X2 5.05 -Y2 5.62 -Color $orange -Weight 1.4 | Out-Null
MatrixIcon -Page $page -X 5.25 -Y 5.08 -Rows 6 -Cols 6 -Cell 0.105 -Palette @("RGB(46,125,50)","RGB(129,199,132)","RGB(255,255,255)","RGB(67,160,71)")
Text -Page $page -X 5.12 -Y 4.78 -W 1.35 -H 0.20 -Text "A_lag mean" -FontSize 8 -Color $ink | Out-Null
MatrixIcon -Page $page -X 0.90 -Y 4.12 -Rows 5 -Cols 5 -Cell 0.105 -Palette @("RGB(255,255,255)","RGB(248,113,113)","RGB(252,165,165)")
Text -Page $page -X 0.75 -Y 3.88 -W 1.30 -H 0.18 -Text "A0 residual" -FontSize 7.5 -Color $ink | Out-Null
PlusIcon -Page $page -X 2.32 -Y 4.30
Text -Page $page -X 2.66 -Y 4.30 -W 1.18 -H 0.20 -Text "DAG + sparse" -FontSize 8 -Color $ink -HAlign 0 | Out-Null
Arrow -Page $page -X1 4.00 -Y1 4.40 -X2 5.10 -Y2 4.40 -Color $green -Weight 1.2 -Dashed $true -Label "top-k" | Out-Null
MatrixIcon -Page $page -X 5.25 -Y 3.95 -Rows 6 -Cols 6 -Cell 0.105 -Palette @("RGB(255,255,255)","RGB(34,197,94)","RGB(187,247,208)")
Text -Page $page -X 5.10 -Y 3.78 -W 1.42 -H 0.18 -Text "causal gate" -FontSize 7.5 -Color $ink | Out-Null

# Panel C: feature extraction / hyperbolic graph learning.
Box -Page $page -X 7.05 -Y 3.75 -W 7.00 -H 3.22 -Text "" -Fill "RGB(255,255,255)" -Line $dash -Weight 1.4 -Dashed $true -Round 0.12 | Out-Null
Text -Page $page -X 11.62 -Y 6.58 -W 2.00 -H 0.28 -Text "(C) Representation" -FontSize 13 -Color $ink | Out-Null
VectorIcon -Page $page -X 7.45 -Y 5.15 -N 7 -Color "RGB(166,76,16)" -W 0.24 -H 0.20
Text -Page $page -X 7.22 -Y 4.82 -W 0.95 -H 0.20 -Text "C" -FontSize 9 -Color $ink | Out-Null
for ($i=0; $i -lt 4; $i++) {
    $x1 = 7.78
    $y1 = 5.25 + $i*0.25
    $x2 = 8.80
    $y2 = 5.10 + (($i+1)%4)*0.25
    Arrow -Page $page -X1 $x1 -Y1 $y1 -X2 $x2 -Y2 $y2 -Color "RGB(110,110,110)" -Weight 0.6 -Dashed $true | Out-Null
}
VectorIcon -Page $page -X 8.92 -Y 5.08 -N 5 -Color $blue -W 0.28 -H 0.23
Text -Page $page -X 8.60 -Y 4.82 -W 1.20 -H 0.20 -Text "HGCN" -FontSize 9 -Color $ink | Out-Null
Arrow -Page $page -X1 9.45 -Y1 5.76 -X2 10.05 -Y2 5.76 -Color $orange -Weight 1.4 | Out-Null
NodeGraphIcon -Page $page -X 10.10 -Y 5.05 -Color $blue
Text -Page $page -X 10.00 -Y 4.82 -W 1.15 -H 0.20 -Text "A_cls graph" -FontSize 8 -Color $ink | Out-Null
Arrow -Page $page -X1 11.18 -Y1 5.76 -X2 11.78 -Y2 5.76 -Color $orange -Weight 1.4 | Out-Null
Box -Page $page -X 11.88 -Y 5.18 -W 1.28 -H 1.02 -Text "Readout" -Fill $softPurple -Line $purple -FontSize 9 | Out-Null
Arrow -Page $page -X1 12.52 -Y1 5.18 -X2 12.52 -Y2 4.55 -Color $green -Weight 1.2 | Out-Null
Box -Page $page -X 11.65 -Y 4.05 -W 1.75 -H 0.36 -Text "z_global" -Fill "RGB(232,223,255)" -Line $purple -FontSize 9 | Out-Null
Box -Page $page -X 8.35 -Y 3.98 -W 2.35 -H 0.54 -Text "Hyperbolic graph embedding" -Fill "RGB(247,243,255)" -Line $purple -FontSize 8.5 | Out-Null
Arrow -Page $page -X1 10.70 -Y1 4.25 -X2 11.65 -Y2 4.25 -Color $green -Weight 1.2 | Out-Null

# Panel D: feature fusion and classification.
Box -Page $page -X 14.30 -Y 3.75 -W 3.15 -H 3.22 -Text "" -Fill "RGB(255,255,255)" -Line $dash -Weight 1.4 -Dashed $true -Round 0.12 | Out-Null
Text -Page $page -X 14.75 -Y 6.45 -W 2.25 -H 0.46 -Text (Join-Lines @("(D) Prototype energy", "and classification")) -FontSize 12 -Color $ink | Out-Null
VectorIcon -Page $page -X 14.70 -Y 5.25 -N 7 -Color $blue -W 0.22 -H 0.18
VectorIcon -Page $page -X 14.70 -Y 4.20 -N 7 -Color $red -W 0.22 -H 0.18
Text -Page $page -X 14.40 -Y 5.95 -W 0.88 -H 0.20 -Text "class 0" -FontSize 7.5 -Color $ink | Out-Null
Text -Page $page -X 14.40 -Y 3.95 -W 0.88 -H 0.20 -Text "class 1" -FontSize 7.5 -Color $ink | Out-Null
PlusIcon -Page $page -X 15.25 -Y 5.02
Arrow -Page $page -X1 15.52 -Y1 5.13 -X2 16.02 -Y2 5.13 -Color $ink -Weight 1.2 | Out-Null
VectorIcon -Page $page -X 16.05 -Y 4.60 -N 9 -Color "RGB(130,130,130)" -W 0.17 -H 0.15
for ($i=0; $i -lt 5; $i++) {
    Arrow -Page $page -X1 16.25 -Y1 (4.72 + $i*0.18) -X2 16.92 -Y2 (5.02 + (($i%2)*0.50)) -Color "RGB(55,65,81)" -Weight 0.55 | Out-Null
}
$nc = $page.DrawRectangle(16.98, 5.22, 17.28, 5.52)
$nc.Text = "NC"
$nc.CellsU("FillForegnd").FormulaU = "RGB(106,168,79)"
$nc.CellsU("LineColor").FormulaU = "RGB(30,41,59)"
Style-Text -Shape $nc -Size 8
$mdd = $page.DrawOval(16.98, 4.28, 17.30, 4.60)
$mdd.Text = "MDD"
$mdd.CellsU("FillForegnd").FormulaU = "RGB(237,125,49)"
$mdd.CellsU("LineColor").FormulaU = "RGB(30,41,59)"
Style-Text -Shape $mdd -Size 7
Arrow -Page $page -X1 13.40 -Y1 4.23 -X2 14.30 -Y2 5.15 -Color $blue -Weight 2.0 | Out-Null

# Cross-panel colored flows.
Arrow -Page $page -X1 6.48 -Y1 4.42 -X2 7.05 -Y2 4.42 -Color $green -Weight 2.0 | Out-Null
Arrow -Page $page -X1 6.48 -Y1 5.62 -X2 7.05 -Y2 5.62 -Color $orange -Weight 2.0 | Out-Null
Arrow -Page $page -X1 12.60 -Y1 7.98 -X2 14.30 -Y2 5.78 -Color $orange -Weight 1.6 -Dashed $true -Label "R + causal gate" | Out-Null

# Loss strip.
Box -Page $page -X 0.55 -Y 1.30 -W 16.90 -H 1.55 -Text "" -Fill "RGB(255,255,255)" -Line $dash -Weight 1.2 -Dashed $true -Round 0.10 | Out-Null
Text -Page $page -X 0.78 -Y 2.50 -W 1.48 -H 0.24 -Text "Training losses" -FontSize 11 -Color $ink -HAlign 0 | Out-Null
Box -Page $page -X 0.90 -Y 1.70 -W 2.75 -H 0.48 -Text "L_cls" -Fill $softRed -Line $red -FontSize 10 | Out-Null
Box -Page $page -X 4.08 -Y 1.70 -W 3.15 -H 0.48 -Text "L_pred + L_DAG + L_sparse + L_smooth" -Fill $softGreen -Line $green -FontSize 8.5 | Out-Null
Box -Page $page -X 7.70 -Y 1.70 -W 2.65 -H 0.48 -Text "L_radius + L_proto_sep" -Fill $softPurple -Line $purple -FontSize 8.8 | Out-Null
Box -Page $page -X 10.82 -Y 1.70 -W 2.55 -H 0.48 -Text "teacher distillation" -Fill $softYellow -Line $orange -FontSize 8.8 | Out-Null
Box -Page $page -X 13.82 -Y 1.70 -W 2.80 -H 0.48 -Text "TensorBoard / t-SNE / result.xlsx" -Fill "RGB(232,248,250)" -Line $blue -FontSize 8.2 | Out-Null
Arrow -Page $page -X1 3.70 -Y1 1.94 -X2 4.08 -Y2 1.94 -Color $ink -Weight 1.0 | Out-Null
Arrow -Page $page -X1 7.25 -Y1 1.94 -X2 7.70 -Y2 1.94 -Color $ink -Weight 1.0 | Out-Null
Arrow -Page $page -X1 10.38 -Y1 1.94 -X2 10.82 -Y2 1.94 -Color $ink -Weight 1.0 | Out-Null
Arrow -Page $page -X1 13.38 -Y1 1.94 -X2 13.82 -Y2 1.94 -Color $ink -Weight 1.0 | Out-Null

Text -Page $page -X 0.55 -Y 0.78 -W 16.90 -H 0.28 -Text "Default path: ALFF/fALFF feature extraction, temporal NTS-NOTEARS causal graph, gated-FC graph construction, HGCN representation learning, and HPEC multi-prototype energy classification." -FontSize 8.0 -Color "RGB(75,85,99)" | Out-Null

$doc.SaveAs($OutputPath)
Write-Host "Visio architecture diagram saved to: $OutputPath"

if (-not $OpenVisio) {
    try {
        $doc.Saved = $true
        $doc.Close()
    }
    catch {
        Write-Warning "The diagram was saved, but Visio did not close the document cleanly: $($_.Exception.Message)"
    }
    try {
        $visio.Quit()
    }
    catch {
        Write-Warning "The diagram was saved, but Visio did not quit cleanly: $($_.Exception.Message)"
    }
}
