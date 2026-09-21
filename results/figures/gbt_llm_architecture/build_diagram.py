"""Draw the GBT-LLM architecture diagram (encoder and decoder).

Builds the figure from primitive boxes, labels and arrows and writes it in three
formats next to this script: gbt_llm_architecture.svg, an editable
gbt_llm_architecture.drawio (mxGraph XML for diagrams.net) and a 2x-scaled
gbt_llm_architecture.png. Requires cairosvg. Every element is appended both to
the SVG string and to the draw.io tree, so the two files stay identical.
Coordinates are in pixels on a 1600x880 canvas.

Usage:
    python results/figures/gbt_llm_architecture/build_diagram.py
"""
from pathlib import Path
import html, math, xml.etree.ElementTree as ET
import cairosvg

OUT = Path(__file__).resolve().parent
W,H=1600,880
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/>']
mx=ET.Element('mxfile',host='app.diagrams.net',version='24.7.17')
d=ET.SubElement(mx,'diagram',name='GBT-LLM Architecture',id='gbt-llm')
m=ET.SubElement(d,'mxGraphModel',dx=str(W),dy=str(H),grid='1',gridSize='10',page='1',pageWidth=str(W),pageHeight=str(H))
root=ET.SubElement(m,'root'); ET.SubElement(root,'mxCell',id='0'); ET.SubElement(root,'mxCell',id='1',parent='0')
counter=1  # running id of draw.io cells; ids 0 and 1 are the root cells
def cell(value,style,x,y,w,h):
 """Add a vertex cell (x, y, width, height) to the draw.io model."""
 global counter
 counter+=1
 c=ET.SubElement(root,'mxCell',id=str(counter),value=value,style=style,vertex='1',parent='1')
 ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
def label(x,y,w,h,text,size=17,bold=False,italic=False):
 """Centre (possibly multi-line) text in the box (x, y, w, h) in both outputs."""
 lines=text.split('\n')
 for i,line in enumerate(lines):
  yy=y+h/2+(i-(len(lines)-1)/2)*size*1.2
  svg.append(f'<text x="{x+w/2}" y="{yy}" text-anchor="middle" dominant-baseline="central" font-family="Arial" font-size="{size}" font-weight="{"bold" if bold else "normal"}" font-style="{"italic" if italic else "normal"}">{html.escape(line)}</text>')
 cell(text,'text;html=0;align=center;verticalAlign=middle;whiteSpace=wrap;strokeColor=none;fillColor=none;fontFamily=Arial;fontSize='+str(size)+';fontStyle='+str((1 if bold else 0)+(2 if italic else 0))+';',x,y,w,h)
def box(x,y,w,h,text='',fill='#ffffff',stroke='#000000',dash=False,size=17):
 """Draw a rectangle with an optional centred label."""
 svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"'+(' stroke-dasharray="9 6"' if dash else '')+'/>')
 cell('',f'rounded=0;whiteSpace=wrap;html=0;fillColor={fill};strokeColor={stroke};strokeWidth=1.6;dashed={int(dash)};',x,y,w,h)
 if text: label(x,y,w,h,text,size)
def line(points,dash=False,arrow=True,color='#000000',width=1.6):
 """Draw a polyline through points, with an arrow head at the last point if arrow is set."""
 global counter
 svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="{width}"'+(' stroke-dasharray="7 5"' if dash else '')+'/>')
 if arrow:
  # Arrow head: a 10 px long, 8 px wide triangle aligned with the last segment.
  x,y=points[-1]; a,b=points[-2]; angle=math.atan2(y-b,x-a)
  pts=[(x,y),(x-10*math.cos(angle)+4*math.sin(angle),y-10*math.sin(angle)-4*math.cos(angle)),(x-10*math.cos(angle)-4*math.sin(angle),y-10*math.sin(angle)+4*math.cos(angle))]
  svg.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in pts)+f'" fill="{color}"/>')
 counter+=1
 c=ET.SubElement(root,'mxCell',id=str(counter),edge='1',parent='1',style=f'html=0;endArrow={"block" if arrow else "none"};endFill=1;strokeColor={color};strokeWidth={width};dashed={int(dash)};')
 g=ET.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
 for role,p in [('sourcePoint',points[0]),('targetPoint',points[-1])]: ET.SubElement(g,'mxPoint',x=str(p[0]),y=str(p[1]),attrib={'as':role})
 arr=ET.SubElement(g,'Array',attrib={'as':'points'})
 for a,b in points[1:-1]: ET.SubElement(arr,'mxPoint',x=str(a),y=str(b))
def graph(x,y):
 """Draw the 5x5 grid graph icon (nodes 15 px apart) for the predicted graph W."""
 for r in range(5):
  for c in range(5):
   if c<4: line([(x+c*15,y+r*15),(x+(c+1)*15,y+r*15)],arrow=False,width=.8)
   if r<4: line([(x+c*15,y+r*15),(x+c*15,y+(r+1)*15)],arrow=False,width=.8)
 for r in range(5):
  for c in range(5):
   xx,yy=x+c*15,y+r*15
   svg.append(f'<circle cx="{xx}" cy="{yy}" r="2.3" fill="#3232ef"/>')
   cell('','ellipse;fillColor=#3232ef;strokeColor=none;',xx-2.3,yy-2.3,4.6,4.6)
def context(y):
 """Draw the decoded neighbour blocks I (top-left), J (top) and K (left) with the red causal-context outline; y is the top edge."""
 label(38,y-32,190,25,'Decoded reference blocks',15)
 box(60,y,65,55,'I\n8 × 8','#e4d4df',size=15)
 box(135,y,65,55,'J (top)\n8 × 8','#e2d8ee',size=15)
 box(60,y+65,65,55,'K (left)\n8 × 8','#d9ead3',size=15)
 line([(53,y+125),(53,y-6),(207,y-6),(207,y+61),(132,y+61),(132,y+125),(53,y+125)],arrow=False,color='#ff0000',width=2.2)

label(0,10,W,30,'GBT-LLM: graph-based transform with an LLM entropy prior',23,True)
box(265,65,1275,365,stroke='#3535ff',dash=True)
box(265,490,1275,300,stroke='#3535ff',dash=True)
label(1410,395,115,25,'Encoder',21,True,True)
label(1410,751,115,25,'Decoder',21,True,True)


box(420,95,165,60,'Reconstructed block\nx̂ (round / clip)',size=16)
box(680,95,165,60,'Inverse graph\nFourier transform',size=16)
box(935,95,155,60,'De-quantization\nŷ = Δq',size=16)
line([(935,125),(845,125)])
line([(680,125),(585,125)])
line([(420,125),(295,125),(295,275),(220,275)])
label(297,155,115,40,'Reconstructed\ncontext update',13)

box(135,195,95,65,'Current\nblock x\n8 × 8',fill='#fff2cc',size=16)
box(420,195,165,65,'Graph Fourier\ntransform\ny = Uᵀx',size=16)
box(680,195,165,65,'Quantization\nq = round(y / Δ)',size=16)
box(1320,195,170,65,'Arithmetic\nrange encoder',size=17)
line([(230,227),(420,227)])
line([(585,227),(680,227)])
line([(845,227),(1320,227)])
label(970,201,220,20,'Quantized coefficients q',14)
line([(880,227),(880,173),(1012,173),(1012,155)])

context(290)
box(315,330,170,60,'Causal context +\nin-context support',size=16)
box(520,330,160,60,'GBT-ICL\nmeta-learner',fill='#c9daf8',size=17)
graph(721,330)
label(690,396,130,20,'Predicted graph W',13)
box(825,330,165,60,'Graph Laplacian\nL = D − W = UΛUᵀ',size=16)
box(1100,330,210,60,'LLM coefficient prior\nDistilGPT-2 + LoRA',fill='#e4d7ef',size=16)
line([(207,350),(315,350)])
line([(485,360),(520,360)])
line([(680,360),(714,360)])
line([(788,360),(825,360)])
line([(905,330),(905,287),(502,287),(502,260)],dash=True)
label(575,266,70,20,'Basis U',13)
line([(905,287),(905,178),(762,178),(762,155)],dash=True)
line([(990,360),(1100,360)],dash=True)
label(1000,334,88,20,'Spectrum Λ',13)
line([(1205,330),(1205,282),(1405,282),(1405,260)],dash=True)
label(1225,289,245,20,'P(q_k | q<k, Λ)',14)
line([(1060,227),(1060,307),(1150,307),(1150,330)],dash=True)
label(912,290,143,30,'Coefficient history',13)


line([(1490,227),(1560,227),(1560,530),(1490,530)])
box(1395,446,180,29,'Compressed bitstream',size=14)
box(1320,500,170,60,'Arithmetic\nrange decoder',size=17)
box(1060,500,155,60,'De-quantization\nŷ = Δq',size=16)
box(795,500,170,60,'Inverse graph\nFourier transform',size=16)
box(445,500,195,60,'Reconstructed block\nx̂ = Uŷ (round / clip)',size=16)
line([(1320,530),(1215,530)])
label(1230,506,65,18,'q',14)
line([(1060,530),(965,530)])
line([(795,530),(640,530)])
box(55,500,165,65,'Reconstructed\ncurrent block\n8 × 8',size=16)
line([(445,530),(220,530)])
line([(315,530),(315,589),(220,589),(220,646)])
label(315,561,130,25,'Context update',13)
context(650)
box(315,680,170,60,'Causal context +\nin-context support',size=16)
box(520,680,160,60,'GBT-ICL\nmeta-learner',fill='#c9daf8',size=17)
graph(721,680)
label(690,746,130,20,'Predicted graph W',13)
box(825,680,165,60,'Graph Laplacian\nL = D − W = UΛUᵀ',size=16)
box(1100,680,210,60,'LLM coefficient prior\nDistilGPT-2 + LoRA',fill='#e4d7ef',size=16)
line([(207,710),(315,710)])
line([(485,710),(520,710)])
line([(680,710),(714,710)])
line([(788,710),(825,710)])
line([(905,680),(905,610),(880,610),(880,560)],dash=True)
label(903,608,85,25,'Basis U',13)
line([(990,710),(1100,710)],dash=True)
label(1000,683,88,20,'Spectrum Λ',13)
line([(1205,680),(1205,610),(1405,610),(1405,560)],dash=True)
label(1252,616,235,25,'Symbol probabilities',14)
line([(1260,530),(1260,584),(1145,584),(1145,680)],dash=True)
label(986,633,155,28,'Decoded history q<k',13)

line([(285,825),(345,825)])
label(350,811,130,28,'Signal / data',14)
line([(510,825),(570,825)],dash=True)
label(575,811,240,28,'Basis / model conditioning',14)
label(35,846,1530,22,'8 × 8 blocks processed in raster order • Shared model weights • Graph and probabilities regenerated at the decoder',14)
svg.append('</svg>')
data=''.join(svg)
(OUT/'gbt_llm_architecture.svg').write_text(data,encoding='utf-8')
ET.indent(mx)
ET.ElementTree(mx).write(OUT/'gbt_llm_architecture.drawio',encoding='utf-8',xml_declaration=True)
cairosvg.svg2png(bytestring=data.encode(),write_to=str(OUT/'gbt_llm_architecture.png'),scale=2)
print(OUT)
