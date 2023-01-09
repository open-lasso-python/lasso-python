OVERHEAD_STRING = """
<html>
  <title>3D Beta Embedding</title>
<head>
  <meta charset="utf-8">
  <style>
    :root{--background:#cfd8dc;--menue-background:#ffffff;--menue-option-background:#ffffff;--menue-option-text-color:#000000;--menue-option-active:#e2f0fb;--content-background:#ffffff;--content-border:1px solid #9ea7aa;--content-shadow:0 4px 8px 0 rgba(0, 0, 0, 0.2),0 6px 20px 0 rgba(0, 0, 0, 0.19);--plot-option-color:#455a64}body{font-family:Lato,sans-serif;background-color:var(--background)}.inputHidden{padding:0;width:0;height:0;display:none;transition:1s}.inputSelected{padding:0 8px 8px 16px;padding-top:15px;display:block;overflow:auto;transition:1s}.menuOption{padding:8px 8px 8px 32px;background-color:var(--menue-option-background)}.navClosed{margin-top:15px;position:fixed;left:1%;transition:.2s;display:flex;min-height:100vh}.navOpen{margin-top:15px;position:fixed;transition:.2s;display:flex;min-height:100vh}.sidenav{height:100%;width:0;position:fixed;z-index:1;top:0;left:0;background-color:var(--menue-background);overflow-x:hidden;transition:.2s;padding-top:60px}.sidenav a{text-decoration:none;font-size:20px;color:var(--menue-option-text-color);display:block;opacity:.7;cursor:pointer}.slider{-webkit-appearance:none;width:90%;background:#137a94;height:5px;border-radius:5px;outline:0}.slider:hover{box-shadow:0 0 .5px .5px #2dceda}.slider::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:15px;height:15px;background:#137a94;border-radius:20px;cursor:pointer}.slider::-moz-range-thumb{width:15px;height:15px;background:#137a94;border-radius:20px;cursor:pointer}.sidenav input[type=text]{background-color:var(--menue-background);border-color:#137a94;width:100%}input[type=text]:focus{outline:1px solid #1995b4}.sidenav a:hover{opacity:.9;background-color:var(--menue-option-active)}.sidenav .closebtn{position:absolute;top:0;right:10px;font-size:36px;margin-left:50px;z-index:100}.plotDiv{border-radius:5px;box-shadow:var(--content-shadow);border:var(--content-border);position:relative;background-color:var(--content-background)}.plotDiv p{position:absolute;top:5px;right:15px}#plotOptions{position:absolute;top:5px;left:15px;font-size:30px;cursor:pointer}#downloadPlot{position:absolute;left:0;color:var(--plot-option-color);text-decoration:underline;text-decoration-thickness:3px;text-underline-offset:5px}#resetPlot{position:absolute;margin-left:30px;color:var(--plot-option-color);font-size:35px}#sizeDrag{cursor:col-resize;width:2%}#dragLineDiv{width:0;border:var(--content-border);margin-left:45%;margin-right:49%}.imgDiv{border-radius:5px;box-shadow:var(--content-shadow);border:var(--content-border);display:flex;justify-content:center;align-items:center;background-color:var(--content-background)}.alignedIMG{width:99%;height:auto}.imgDiv :is(p,h2){margin-top:0;padding-left:15px;display:inline-block;vertical-align:middle}.noImgDescr{position:relative;display:none}.traceContainerClass{float:left;padding-bottom:5px}.traceContainerClass :is(p,canvas){float:left}.traceContainerClass p{font-size:16px;padding-left:5px;padding-right:50px;color:var(--menue-option-text-color);margin-top:15px;opacity:.7}.traceContainerClass p:hover{opacity:.9}.traceContainerClass canvas{border-radius:24px;margin-bottom:15px;margin-top:5px}.traceContainerClass canvas:hover{box-shadow:0 0 3px 2px #12678f;transition:.4s}.colorwheel{border-radius:128px!important;border-color:#f1f1f1;position:relative;z-index:10;top:0;left:0}
  </style>
</head>
<body onresize="resizeContents()">
  <div id="mySidenav" class="sidenav">
    <a href="javascript:void(0)" class="closebtn" onclick="closeNav()">&times;</a>
    <div class="menuOption">
      <a href="#" id="imgDirBtn" onclick="showInputField('imgDirDiv', 'imgDirBtn')">Image Directory</a>
      <div id="imgDirDiv" class="inputHidden">
        <input id="imageDir" type="text" onchange="updateIMGPath(event)"
          placeholder="Enter path to img directory">
      </div>
    </div>
    <div class="menuOption">
      <a href="#" id="imgEndBtn" onclick="showInputField('imgEndDiv', 'imgEndBtn')">Image Filetype</a>
      <div id="imgEndDiv" class="inputHidden">
        <input id="imageEnd" type="text" onchange="updateIMGEnd(event)"
          placeholder="Enter img type ending">
      </div>
    </div>
    <div class="menuOption">
      <a href="#" id="pointSliderBtn" onclick="showInputField('sliderDiv', 'pointSliderBtn')">Point Size</a>
      <div id="sliderDiv" class="inputHidden"><input id="slider_pt_size" type="range" class="slider"></div>
    </div>
    <div class="menuOption">
      <a href="#" id="borderSliderBtn" onclick="showInputField('borderSliderDiv', 'borderSliderBtn')">Border Size</a>
      <div id="borderSliderDiv" class="inputHidden"><input id="slider_border_size" type="range" class="slider"></div>
    </div>
    <div class="menuOption">
      <a href="#" id="traceColorBtn" onclick="showInputField('traceColorDiv', 'traceColorBtn')">Point Color</a>
      <div id="traceColorDiv" class="inputHidden"></div>
    </div>
    <?xml version="1.0" encoding="UTF-8" standalone="no"?>
    <svg id="lassoLogo" style="position: absolute; top: 15px; left: 32px;" height="50px" preserveAspectRatio="none" data-name="Ebene 1" version="1.1" viewBox="0 0 171.61 53.835" xmlns="http://www.w3.org/2000/svg" xmlns:cc="http://creativecommons.org/ns#" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:xlink="http://www.w3.org/1999/xlink">
      <metadata>
      <rdf:RDF>
      <cc:Work rdf:about="">
      <dc:format>image/svg+xml</dc:format>
      <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
      <dc:title>Zeichenfläche 1</dc:title>
      </cc:Work>
      </rdf:RDF>
      </metadata>
      <defs>
      <style>.cls-1{fill:url(#a);}.cls-2{fill:#328ccc;}</style>
      <linearGradient id="a" x1="55.32" x2="149.78" y1="73.4" y2="-21.07" gradientTransform="translate(-12 -.18)" gradientUnits="userSpaceOnUse">
      <stop stop-color="#006eb7" offset="0"/>
      <stop stop-color="#1178be" offset=".26"/>
      <stop stop-color="#2987c8" offset=".72"/>
      <stop stop-color="#328ccc" offset="1"/>
      </linearGradient>
      <linearGradient id="b" x1="55.32" x2="149.78" y1="73.4" y2="-21.07" gradientTransform="translate(-21,-9.18)" gradientUnits="userSpaceOnUse" xlink:href="#a"/>
      </defs>
      <title>Zeichenfläche 1</title>
      <path class="cls-1" d="m153 1h-38.39-40.2a8.5 8.5 0 1 0 0 17h4.94a7.09 7.09 0 0 1 0 14.17h-39.53a2.5 2.5 0 1 1 0-5h12a3.89 3.89 0 0 0 3.65-5.22l-6.22-17.13a5.64 5.64 0 0 0-10.62 0l-8.95 24.59a4.25 4.25 0 0 1-4 2.8h-17.62a4.25 4.25 0 0 1-4.23-4.25v-26.92h-2.83v26.92a7.08 7.08 0 0 0 7.06 7.08h17.65a7.08 7.08 0 0 0 6.64-4.66l8.95-24.64a2.82 2.82 0 0 1 5.3 0l6.23 17.26a1.07 1.07 0 0 1-1 1.43h-12a5.31 5.31 0 0 0 0 10.62h74.82a9.915 9.915 0 0 0 0-19.83h-4.94a5.67 5.67 0 1 1 0-11.34h33.93a16.92 16.92 0 1 0 9.4-2.87zm-43.33 17h4.94a7.09 7.09 0 0 1 0 14.17h-28.4a9.91 9.91 0 0 0-6.9-17h-4.9a5.67 5.67 0 1 1 0-11.34h29a8.49 8.49 0 0 0 6.29 14.17zm43.33 14.17a14.16 14.16 0 0 1-0.77-28.3h0.79a14.17 14.17 0 0 1 0 28.34z" fill="url(#b)"/>
    </svg>
  </div>
  <span style="font-size: 30px; cursor: pointer;" onclick="openNav()">&#9776; <span style="font-size: 25px;">Options</span>
  </span>
  <div id="content" class="navClosed">
    <div id="plotDiv" class="plotDiv">
      <p id="plotInfo"></p>
      <div id="plotOptions">
        <span id="downloadPlot" onclick="downloadPlot()" title="Download Plot as png"><a>&#8595;</a></span>
        <span id="resetPlot" title="Reset Plot"><a onclick="resetRenderer()">&#8634;</a></span>
      </div>
    </div>
    <div id="sizeDrag">
      <div id="dragLineDiv"></div>
    </div>
    <div id="imgDiv" class="imgDiv">
      <div><p id="messageP">Hover over points to see the corresponding image </p></div>
      <div style="display: none;"><img id="hoverIMG" class="alignedIMG"></div>
      <div id="noImgDescr" class="noImgDescr">
        <h2>No Image</h2>
        <br>
        <p>You can set the image directory in the options tab.</p>
        <p>Every image must have a number as name which corresponds to the run id.</p>
      </div>
    </div>
  </div>
"""  # noqa: E501

TRACE_STRING = """
  {_traceNr_} =
    {{
      name: '{_name_}',
      color: '{_color_}',
      border: '#555555',
      text: {_runIDs_},
      x: {_x_},
      y: {_y_},
      z: {_z_},
      vertices: []
    }},
"""

CONST_STRING = """
  {_three_min_}
  <script>
    path = "{_path_str_}"
    imgType = "png"
    const runIDs = {_runIdEntries_}
"""

SCRIPT_STRING = """
    const pointSlider = document.getElementById("slider_pt_size")
    const borderSlider = document.getElementById("slider_border_size")
    const plotDiv = document.getElementById("plotDiv")
    const imgDiv = document.getElementById("imgDiv")
    const drag = document.getElementById("sizeDrag")

    let plotDivMult = 0.7
    let sideNavWidthMult = 0.15

    const camera = new THREE.PerspectiveCamera(40, 2, 0.1, 10000)//(fov, aspect, near, far);
    const cameraStartZ = 500
    camera.position.z = cameraStartZ

    const renderer = new THREE.WebGLRenderer({alpha: true, preserveDrawingBuffer: true})

    const scene = new THREE.Scene()
    const renderObjects = new THREE.Group()
    const centeredGroup = new THREE.Group()
    centeredGroup.add(renderObjects)
    scene.add(centeredGroup)



    mouseProps = {
      start_x : 0,
      start_y : 0,
      is_middle_down : false,
      is_left_down : false,
      is_right_down : false,
    }

    const dragFunc = (e) => {

      let widthModi = parseFloat(document.getElementById("mySidenav").style.width)
      let userWidth = window.innerWidth - widthModi

      const startWidth = parseFloat(plotDiv.style.width)
      document.selection ? document.selection.empty() : window.getSelection().removeAllRanges()
      const newPlotWidth = (e.pageX - drag.offsetWidth / 2 - widthModi)

      const newPlotDivMult = newPlotWidth / userWidth

      if(!(newPlotDivMult < 0.1) && !(newPlotDivMult > 0.9)) {
        plotDivMult = newPlotDivMult
        resizeContents()
      }
    }

    function initDocument(){
      pointSlider.min = 5
      pointSlider.max = 35
      pointSlider.step = 1
      pointSlider.value = 20

      borderSlider.min = 1
      borderSlider.max = 10
      borderSlider.step = 1
      borderSlider.value = 3

      document.getElementById("mySidenav").style.width = 0

      const plotDivWidth = window.innerWidth * plotDivMult
      const imgDivWidth = window.innerWidth * (1 - plotDivMult - 0.02)

      plotDiv.style.width = plotDivWidth
      imgDiv.style.width = imgDivWidth

      plotDiv.style.height = 0.9 * window.innerHeight
      imgDiv.style.height = 0.9 * window.innerHeight
      drag.children[0].style.height = parseFloat(plotDiv.style.height)

      renderer.setSize(plotDivWidth, Math.min(0.9*window.innerHeight, plotDivWidth), false)
      plotDiv.appendChild(renderer.domElement)

      if(path !== ""){
        document.getElementById("imageDir").value = path
      }
      document.getElementById("imageEnd").value = imgType

      drag.addEventListener("mousedown", () =>  {
        document.addEventListener("mousemove", dragFunc)
      })
      document.addEventListener("mouseup", () => {
        document.removeEventListener("mousemove", dragFunc)
      })
      traceList.forEach(trace =>{
        for (let n=0; n < trace.x.length; n++){
          trace.vertices.push(trace.x[n], trace.y[n], trace.z[n])
        }
      })
    }

    function resizeContents(){
      const sideNav = document.getElementById("mySidenav")
      if(plotDiv.parentElement.className === "navOpen"){
        sideNav.style.width = Math.min(250, window.innerWidth*sideNavWidthMult) + "px";
        document.getElementById("lassoLogo").style.width = parseFloat(sideNav.style.width) * 0.6 - 30
      }

      let sideNavWidth = parseFloat(sideNav.style.width)
      let userWidth = window.innerWidth - sideNavWidth
      let contentBound = window.innerWidth * 0.01

      plotDiv.parentElement.style.left = sideNavWidth + contentBound
      plotDiv.style.width = userWidth * plotDivMult + 'px'
      imgDiv.style.width = userWidth * (1-plotDivMult) - 2*contentBound + 'px';
      plotDiv.style.height = 0.9 * window.innerHeight
      imgDiv.style.height = 0.9 * window.innerHeight
      plotDiv.removeChild(renderer.domElement)
      renderer.setSize(userWidth*plotDivMult, Math.min(0.9*window.innerHeight, userWidth*plotDivMult), false)
      plotDiv.appendChild(renderer.domElement)
      drag.children[0].style.height = parseFloat(plotDiv.style.height)
    }

    function openNav(){
      const sideNav = document.getElementById("mySidenav")
      sideNav.style.boxShadow = "var(--content-shadow)"
      document.getElementById("content").setAttribute("class", "navOpen");
      resizeContents()
    }
    function closeNav(){
      const sideNav = document.getElementById("mySidenav")
      sideNav.style.width = "0";
      sideNav.style.boxShadow = ""
      document.getElementById("content").setAttribute("class", "navClosed");
      hideAll();
      resizeContents();
    }

    function showInputField(div, aClicked){
      hideAll()
      document.getElementById(div).setAttribute("class", "inputSelected")
      document.getElementById(aClicked).setAttribute("onclick", `hideInputField('${div}', '${aClicked}')`)
      document.getElementById(aClicked).style.background = "var(--menue-option-active)"
    }

    function hideInputField(div, aClicked){
      document.getElementById(div).setAttribute("class", "inputHidden")
      document.getElementById(aClicked).setAttribute("onclick", `showInputField('${div}', '${aClicked}')`)
      document.getElementById(aClicked).style.background = "var(--menue-option-background)"
    }

    function hideAll(){
      document.getElementById("sliderDiv").setAttribute('class', 'inputHidden')
      document.getElementById("imgDirDiv").setAttribute('class', 'inputHidden')
      document.getElementById("imgEndDiv").setAttribute('class', 'inputHidden')
      document.getElementById("traceColorDiv").setAttribute('class', 'inputHidden')
      document.getElementById("borderSliderDiv").setAttribute('class', 'inputHidden')
      document.getElementById("pointSliderBtn").setAttribute("onclick", "showInputField('sliderDiv', 'pointSliderBtn')")
      document.getElementById("imgDirBtn").setAttribute("onclick", "showInputField('imgDirDiv', 'imgDirBtn')")
      document.getElementById("imgEndBtn").setAttribute("onclick", "showInputField('imgEndDiv', 'imgEndBtn')")
      document.getElementById("traceColorBtn").setAttribute("onclick", "showInputField('traceColorDiv', 'traceColorBtn')")
      document.getElementById("borderSliderBtn").setAttribute("onclick", "showInputField('borderSliderDiv','borderSliderBtn')")
      Array.from(document.getElementById("mySidenav").children).forEach(node => {
        if(node.childElementCount > 0){
          node.children[0].style.background = "var(--menue-option-background)"
        }
      })
    }

    let mouseDownInWheel = false
    let colorWheelActive = false

    document.addEventListener('mousedown', function(event){
      if (!(event.target.id.slice(0, 5) === "color")){
        if(colorWheelActive){
          document.getElementById("colorwheel").remove()
          colorWheelActive = false
        }
      }
    })

    function openWheel(event){
      if(colorWheelActive){
        document.getElementById("colorwheel").remove()
      }
      const traceId = parseInt(event.target.id.slice(-1))
      const wheel = makeWheel(128)
      wheel.className = "colorwheel"
      wheel.id = "colorwheel"
      wheel.addEventListener('mousedown', function(event){
        mouseDownInWheel = true
        let wheelCanvas = this
        updateMarkerColor(event, wheelCanvas)
      })
      wheel.addEventListener('mousemove', function(event){
        let wheelCanvas = this
        updateMarkerColor(event, wheelCanvas)
      })
      wheel.addEventListener('mouseup', removeColorWheel)
      wheel.addEventListener('mouseout', removeColorWheel)
      event.target.parentElement.appendChild(wheel)
      colorWheelActive = true
    }

    function updateMarkerColor(event, wheelCanvas){
      if (mouseDownInWheel) {
        let traceID = parseInt(event.target.parentElement.id.slice(-1)) ? parseInt(event.target.parentElement.id.slice(-1)) : 0
        console.log(traceID)
        let targetTrace = traceList[traceID]
        let wheelCtx = wheelCanvas.getContext('2d')
        let data = wheelCtx.getImageData(event.offsetX, event.offsetY, 1, 1).data
        let rgba = 'rgba(' + data[0] + ',' + data[1] +
                   ',' + data[2] + ',' + (data[3] / 255) + ')'
        targetTrace.color = rgba
        const markerSize = document.getElementById("slider_pt_size").value
        const borderSize = document.getElementById("slider_border_size").value
        // renderObjects.children = []
        const targetCanvas = event.target.parentElement.children[0]
        createMarker(16, 3, rgba, targetTrace.border, targetCanvas)
        // traceList.forEach(trace => {
          // addPoints(trace, markerSize, borderSize)
        // })
        const newTexture = createMarker(markerSize, borderSize, rgba, targetTrace.border,document.createElement("canvas"))
        renderObjects.children[traceID].material.map.image = newTexture
        renderObjects.children[traceID].material.map.needsUpdate = true


      }
    }

    function removeColorWheel(event){
      if(mouseDownInWheel){
        mouseDownInWheel = false
        let wheelCanvas = this
        this.remove()
        colorWheelActive = false
      }
    }

    function updateIMGPath(){
      path = document.getElementById("imageDir").value
    }

    function updateIMGEnd(){
      imgType = document.getElementById("imageEnd").value
    }

    function createMarker(radius, boundary, color, border, canv){

      const ctx = canv.getContext('2d')
      ctx.canvas.height = radius * 2 + 10
      ctx.canvas.width = radius * 2 + 10
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
      ctx.beginPath();
      ctx.arc(ctx.canvas.width / 2, ctx.canvas.height / 2, radius, 0, Math.PI * 2)
      ctx.fillStyle = color
      ctx.fill()
      ctx.beginPath();
      ctx.arc(ctx.canvas.width / 2, ctx.canvas.height / 2, radius, 0, Math.PI * 2)
      ctx.strokeStyle = border
      ctx.lineWidth = boundary
      ctx.stroke()

      return ctx.canvas
    }

    function createHoverPoint(point, color) {
      const radius = parseFloat(document.getElementById("slider_pt_size").value)+1
      const outerRadius = radius+1+Math.floor(2*radius/3)
      vertices=[]
      vertices.push(point.x, point.y, point.z)

      var geometry = new THREE.BufferGeometry()

      const canv = document.createElement("canvas")
      canv.width = 2 * outerRadius
      canv.height = 2 * outerRadius
      const ctx = canv.getContext('2d')
      const grd = ctx.createRadialGradient(canv.width/2, canv.height/2, 0.5 * radius, canv.width/2, canv.height/2, outerRadius)
      grd.addColorStop(0, color)
      grd.addColorStop(1, "transparent")
      ctx.fillStyle = grd
      ctx.fillRect(0, 0, 2*(outerRadius)+10, 2*(outerRadius)+10)

      const texture = new THREE.CanvasTexture(ctx.canvas)
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3))
      material = new THREE.PointsMaterial({size: outerRadius, sizeAttenuation: false, map: texture, alphaTest: 0.5, transparent: true})
      var particle = new THREE.Points(geometry, material)
      return particle
    }

    function addPoints(trace, radius, boundary){

      var geometry = new THREE.BufferGeometry();

      const newCanv = document.createElement("canvas")
      const texture = new THREE.CanvasTexture(createMarker(radius, boundary, trace.color, trace.border, newCanv))

      geometry.setAttribute('position', new THREE.Float32BufferAttribute(trace.vertices, 3))
      material = new THREE.PointsMaterial({ size: radius, sizeAttenuation: false, map: texture, alphaTest: 0.5, transparent: true })
      var particles = new THREE.Points(geometry, material)
      particles.name =  trace.name
      renderObjects.add(particles)
      renderer.render(scene, camera)
    }


    let hsv2rgb = function(hsv) {
      let h = hsv.hue, s = hsv.sat, v = hsv.val;
      let rgb, i, data = [];
      if (s === 0) {
        rgb = [v,v,v];
      } else {
        h = h / 60;
        i = Math.floor(h);
        data = [v*(1-s), v*(1-s*(h-i)), v*(1-s*(1-(h-i)))];
        switch(i) {
          case 0:
            rgb = [v, data[2], data[0]];
            break;
          case 1:
            rgb = [data[1], v, data[0]];
            break;
          case 2:
            rgb = [data[0], v, data[2]];
            break;
          case 3:
            rgb = [data[0], data[1], v];
            break;
          case 4:
            rgb = [data[2], data[0], v];
            break;
          default:
            rgb = [v, data[0], data[1]];
            break;
        }
      }
      return rgb;
    };

    function clamp(min, max, val)
    {
      if (val < min) return min;
      if (val > max) return max;
      return val;
    }

    function makeWheel(diameter)
    {
      let can = document.createElement('canvas');
      let ctx = can.getContext('2d');
       const divisor = 1.8 // how "zoomed in" the colorwheel is
      can.width = diameter;
      can.height = diameter;
      let imgData = ctx.getImageData(0,0,diameter,diameter);
      let maxRange = diameter / divisor;
      for (let y=0; y<diameter; y++) {
        for (let x=0; x<diameter; x++) {
          let xPos = x - (diameter/2);
          let yPos = (diameter-y) - (diameter/2);

          let polar = pos2polar( {x:xPos, y:yPos} );
          let sat = clamp(0,1,polar.len / ((maxRange/2)));
          let val = clamp(0,1, (maxRange-polar.len) / (maxRange/2) );

          let rgb = hsv2rgb( {hue:polar.ang, sat:sat, val:val} );

          let index = 4 * (x + y*diameter);
          imgData.data[index + 0] = rgb[0]*255;
          imgData.data[index + 1] = rgb[1]*255;
          imgData.data[index + 2] = rgb[2]*255;
          imgData.data[index + 3] = 255;
        }
      }
      ctx.putImageData(imgData, 0,0);
      return can;
    }

    function rad2deg(rad)
    {
      return (rad / (Math.PI * 2)) * 360;
    }

    function pos2polar(inPos)
    {
      let vecLen = Math.sqrt( inPos.x*inPos.x + inPos.y*inPos.y );
      let something = Math.atan2(inPos.y,inPos.x);
      while (something < 0)
        something += 2*Math.PI;

      return { ang: rad2deg(something), len: vecLen };
    }

    function downloadPlot(){
      let downloadLink = document.createElement('a');
      downloadLink.setAttribute('download', 'PointEmbedding.png');
      let canvas = renderer.domElement;
      let dataURL = canvas.toDataURL('image/png');
      let url = dataURL.replace(/^data:image\\/png/,'data:application/octet-stream');
      downloadLink.setAttribute('href',url);
      downloadLink.click();
    }

    initDocument()

    traceList.forEach(trace =>{
      const traceDiv = document.getElementById('traceColorDiv')
      const borderDiv = document.getElementById('borderColorDiv')
      const traceContainDiv = document.createElement('div')
      const traceMarkCanvas = createMarker(16, 3, trace.color, trace.border, document.createElement("canvas"))
      const borderContainDiv = document.createElement('div')
      const borderMarkCanvas = createMarker(16, 3, trace.color, trace.border, document.createElement("canvas"))
      const traceMarkName = document.createElement('p')
      traceMarkName.innerHTML = trace.name
      traceContainDiv.id = "contain" + trace.name
      traceMarkCanvas.id = "color" + trace.name
      traceMarkCanvas.addEventListener('mousedown', openWheel)
      traceDiv.appendChild(traceContainDiv)
      traceContainDiv.appendChild(traceMarkCanvas)
      traceContainDiv.appendChild(traceMarkName)
      traceContainDiv.setAttribute("class", "traceContainerClass")

      addPoints(trace, pointSlider.value, borderSlider.value)
    })

    const box = new THREE.BoxHelper(renderObjects)

    function moveObjectsToBoundingBoxCenter(box) {

      const positions = box.geometry.attributes.position.array
      let xValue = 0
      let yValue = 0
      let zValue = 0
      positions.forEach((pos, ind) => {
        switch(ind%3){
          case 0:
            xValue += pos
            break;
          case 1:
            yValue += pos
            break;
          case 2:
            zValue += pos
        }
      })
      renderObjects.translateX(xValue/-8)
      renderObjects.translateY(yValue/-8)
      renderObjects.translateZ(zValue/-8)

    }

    moveObjectsToBoundingBoxCenter(box)

    pointSlider.oninput = event => {
      const marker_size = parseFloat(event.target.value)
      renderObjects.children = []
      traceList.forEach(trace =>{
        addPoints(trace, marker_size, borderSlider.value)
      })
    }

    borderSlider.oninput = event => {
      const marker_border = parseFloat(event.target.value)
      renderObjects.children = []
      traceList.forEach(trace => {
        addPoints(trace, pointSlider.value, marker_border)
      })
    }

    class PickHelper {
      constructor() {
        this.hitIndex = null
        this.pickedObject = null
      }
      pick(normalizedPosition, scene, camera) {

        // pick depending on point size and scale
        const hitRadius = parseFloat(document.getElementById("slider_pt_size").value) * 0.3 / centeredGroup.scale.x
        const raycaster = new THREE.Raycaster()
        raycaster.params.Points.threshold = hitRadius
        raycaster.setFromCamera(normalizedPosition, camera)
        // get the list of objects the ray intersected
        const intersectedObjects = raycaster.intersectObjects(renderObjects.children)

        if (intersectedObjects.length) {
          // we pick the first object, as it is the closest
          this.hitIndex = intersectedObjects[0].index
          this.pickedObject = intersectedObjects[0].object
        } else {
          this.hitIndex = null
          this.pickedObject = null
        }
      }
    }
    const pickPosition = {x: 0, y: 0};
    const pickHelper = new PickHelper();
    clearPickPosition();

    function getCanvasRelativePosition(event) {
      const rect = renderer.domElement.getBoundingClientRect();
      return {
        x: (event.clientX - rect.left) * renderer.domElement.width  / rect.width,
        y: (event.clientY - rect.top ) * renderer.domElement.height / rect.height,
      };
    }
    function setPickPosition(event) {
      const pos = getCanvasRelativePosition(event);
      pickPosition.x = (pos.x / renderer.domElement.width ) *  2 - 1;
      pickPosition.y = (pos.y / renderer.domElement.height) * -2 + 1;
    }

    function imageExists(imgPath){
      let http = new XMLHttpRequest()
      http.open("HEAD", imgPath, false)
      http.send()
      return http.status != 404;
    }

    const hoverPointInfo = {
      hoverPointSet: false,
      currentTraceIndex: null,
      currentPointIndex: null
    }

    function resetHoverPointInfo(){
      hoverPointInfo.hoverPointSet = false,
      hoverPointInfo.currentTraceIndex = null,
      hoverPointInfo.currentPointIndex = null
    }

    function checkForImageUpdate(){
      if(pickHelper.pickedObject){
        // first we select the trace
        let traceIndex
        let validTraceIndex = false
        for(let i = 0; i<traceList.length; i++){
          if (traceList[i].name === pickHelper.pickedObject.name){
            traceIndex = i
            i = traceList.length
            validTraceIndex = true
          }
        }
        if(validTraceIndex){
          document.getElementById("messageP").parentElement.setAttribute("style" , "display:none")

          const runID = runIDs[traceIndex][pickHelper.hitIndex]
          const img = document.getElementById("hoverIMG")
          img.onerror = function() {
            document.getElementById("noImgDescr").style.display = "block"
            img.parentElement.style.display = "none"
          }
          img.onload = function() {
            document.getElementById("noImgDescr").style.display = "none"
            img.parentElement.style.display = "block"
          }
          img.src = path+`/${runID}.`+imgType
          let infoP = document.getElementById("plotInfo")
          infoP.innerHTML = `${traceList[traceIndex].text[pickHelper.hitIndex]}`

          if(!hoverPointInfo.hoverPointSet || (hoverPointInfo.currentTraceIndex != traceIndex) || (hoverPointInfo.currentPointIndex != pickHelper.hitIndex)){
            if(hoverPointInfo.hoverPointSet){
              renderObjects.children.pop()
            }
            renderObjects.add(createHoverPoint({
              x: traceList[traceIndex].x[pickHelper.hitIndex],
              y: traceList[traceIndex].y[pickHelper.hitIndex],
              z: traceList[traceIndex].z[pickHelper.hitIndex],
            }, traceList[traceIndex].color))
            hoverPointInfo.hoverPointSet = true
            hoverPointInfo.currentTraceIndex = traceIndex
            hoverPointInfo.currentPointIndex = pickHelper.hitIndex
          }
        }
      } else if(hoverPointInfo.hoverPointSet){
        renderObjects.children.pop()
        resetHoverPointInfo()
      }
    }

    function clearPickPosition() {
      pickPosition.x = -100000;
      pickPosition.y = -100000;
    }

    function initMouseProps(event) {
      mouseProps.start_x = event.offsetX
      mouseProps.start_y = event.offsetY
    }

    function mouseClickSelector(event) {
      event.preventDefault()
      if(event.button === 0){
        mouseProps.is_left_down = true
        initMouseProps(event)
      } else if(event.button === 1){
        mouseProps.is_middle_down = true
        initMouseProps(event)
      } else if(event.button === 2){
        mouseProps.is_right_down = true
        initMouseProps(event)
      }
    }

    function toRadians(angle) {
      return angle * (Math.PI / 180);
    }

    function dragMouse(e) {

      let delta_x = e.offsetX - mouseProps.start_x
      let delta_y = e.offsetY - mouseProps.start_y
      mouseProps.start_x = e.offsetX
      mouseProps.start_y = e.offsetY

      if(mouseProps.is_left_down){
        // here we rotate around x and y axis

        var deltaQuaternion = new THREE.Quaternion().setFromEuler(
          new THREE.Euler( toRadians(delta_y), toRadians(delta_x), 0, 'XYZ')
        )
        centeredGroup.quaternion.multiplyQuaternions(deltaQuaternion, centeredGroup.quaternion)

      } else if (mouseProps.is_middle_down) {
        // here we rotate around z axis

        const normPos = getCanvasRelativePosition(event)
        if(normPos.x > parseFloat(plotDiv.style.width) / 2){
          delta_y *= -1
        }
        if(normPos.y < parseFloat(plotDiv.style.height)/2){
          delta_x *= -1
        }
        let rotationRate = (delta_x + delta_y)

        var deltaQuaternion = new THREE.Quaternion().setFromEuler(
          new THREE.Euler( 0, 0, toRadians(rotationRate), 'XYZ')
        )
        centeredGroup.quaternion.multiplyQuaternions(deltaQuaternion, centeredGroup.quaternion)

      } else if (mouseProps.is_right_down) {
        centeredGroup.position.x += delta_x
        centeredGroup.position.y -= delta_y
      }
    }

    function mouseUpHandler(event) {
      if(event.button === 0){
        mouseProps.is_left_down = false
      } else if(event.button === 1){
        mouseProps.is_middle_down = false
      } else if(event.button === 2){
        mouseProps.is_right_down = false
      }
    }


    function onMouseScroll(event) {
      event.preventDefault()
      let deltaY = event.deltaY
      let dirMult = 1

      if(deltaY < 0){
        deltaY *= -1
        dirMult = -1
      }

      const newScale = 0.1 * centeredGroup.scale.x * Math.exp(3/-10)
      centeredGroup.scale.addScalar((newScale * dirMult))
    }

    function resetMouseProps(event) {
      mouseProps.is_left_down = false
      mouseProps.is_middle_down = false
      mouseProps.is_right_down = false
    }

    function resetRenderer() {
      centeredGroup.quaternion.set(0, 0, 0, 1)
      centeredGroup.scale.set(1, 1, 1)
      centeredGroup.position.set(0, 0, 0)
    }

    // events for hover image display
    renderer.domElement.addEventListener('mousemove', setPickPosition);
    renderer.domElement.addEventListener('mouseout', clearPickPosition);
    renderer.domElement.addEventListener('mouseleave', clearPickPosition);

    //events for screen movement
    plotDiv.addEventListener('mousemove', dragMouse)
    plotDiv.addEventListener('mousedown', mouseClickSelector)
    plotDiv.addEventListener('mouseup', mouseUpHandler)
    plotDiv.addEventListener('wheel', onMouseScroll)
    plotDiv.addEventListener('contextmenu', function(event){
      event.preventDefault()
    })
    plotDiv.addEventListener('mouseout', resetMouseProps)

    window.onresize = () => {
      resizeContents()
    }

    function action(time){

      renderer.render(scene, camera)

      requestAnimationFrame(action)
      pickHelper.pick(pickPosition, scene, camera);
      checkForImageUpdate()
    }

    requestAnimationFrame(action)
  </script>
</body>
</html>
"""  # noqa: E501
