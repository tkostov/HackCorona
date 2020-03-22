// the API for Infections (no filtering for now!)
let INFECTIONS_DAY_PARAM = 'http://ec2-3-122-224-7.eu-central-1.compute.amazonaws.com:8080/lk_infections?days=_DAY_&factor=_FACTOR_';

function callAPI(method, page, params, onloadF) {
	var xhr = new XMLHttpRequest();
	xhr.open(method, page, true);
	xhr.setRequestHeader('Content-type', 'application/json');
	xhr.onloadFunction = onloadF; // save it for later!
	
	xhr.onload = function() {
		 if (xhr.status != 200) { // analyze HTTP status of the response
		} else { // show the result
			xhr.onloadFunction();
		}};

	xhr.send();
}


function getMax(p) {
var max = 0;
for(var i = 0; i < p.length; i++){
   var v = p[i][2];
   if (max < v) max = v;						
}
return max;
}



var points2 = [];
var tt = null;
var rs = null;

// heatmapFunctionCases + heatmapFunctionDeads COULD be merged!!!
function heatmapFunctionPercentage() {
	points2 = [];
	var input = {};

	callAPI('GET', INFECTIONS_DAY_PARAM.replace('_DAY_', 0), input,
		function () {
			//log(this);
			tt = this;
			var response = JSON.parse(this.response);
			rs = response;
			for (var i = 0; i < response.length; i++) {
				var p = response[i];
				var g2d = p.geo_point_2d;
				points2.push([g2d[0], g2d[1], p.RelativFall]);
			}
			showHeatmap(points2, getMax(points2));
			if ( document.getElementById("btnPercentage").classList.contains('btn-secondary') && !document.getElementById("btnPercentage").classList.contains('btn-dark'))
				document.getElementById("btnPercentage").classList.toggle('btn-dark');
			if ( document.getElementById("btnAbsolute").classList.contains('btn-dark') )
				document.getElementById("btnAbsolute").classList.toggle('btn-dark');

		});
}

function slide(day) {
	refreshMap(day, );
}

function refreshMap(day, socialDistancing) {
	points2 = [];
	var input = {};

	callAPI('GET', INFECTIONS_DAY_PARAM.replace('_DAY_', day).replace('_FACTOR_', socialDistancing), input,
	function () {
		tt = this;
		var response = JSON.parse(this.response);
		rs = response;
		for (var i = 0; i < response.length; i++) {
			var p = response[i];
			var g2d = p.geo_point_2d;
			points2.push([g2d[0], g2d[1], p.AnzahlFall]);
		}
		var max = getMax(points2);
		// dummy point
		points2.push([16.373703, 10.410084, 1000000]);

		showHeatmap(points2, getMax(points2));
	});
}

function heatmapFunctionAbsolute() {
	points2 = [];

	var input = {};

	callAPI('GET', INFECTIONS_DAY_PARAM.replace('_DAY_', 0), input,
		function () {
			//log(this);
			tt = this;
			var response = JSON.parse(this.response);
			rs = response;
			for (var i = 0; i < response.length; i++) {
				var p = response[i];
				var g2d = p.geo_point_2d;
				points2.push([g2d[0], g2d[1], p.AnzahlFall]);
			}
			showHeatmap(points2, getMax(points2));
			if ( document.getElementById("btnAbsolute").classList.contains('btn-secondary') && !document.getElementById("btnAbsolute").classList.contains('btn-dark') )
				document.getElementById("btnAbsolute").classList.toggle('btn-dark');
			if ( document.getElementById("btnPercentage").classList.contains('btn-dark') )
				document.getElementById("btnPercentage").classList.toggle('btn-dark');
		});
}



var heatmap = null;

function cleanHeatmap() {
	map.removeLayer(heatmap);
	heatmap = null;
}

function showHeatmap(heatdata, maxValue) {
	if (heatmap !== null) {
		cleanHeatmap();
	}

	heatmap = L.heatLayer(heatdata,
	{radius: 35,
	blur: 25,
	gradient: {0.0: 'blue', 0.5: 'lime', 1: 'red'}
	}).addTo(map);

	return heatmap;
}
