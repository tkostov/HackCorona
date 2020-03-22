// the API for Infections (no filtering for now!)
// var INFECTIONS = "http://ec2-3-122-224-7.eu-central-1.compute.amazonaws.com:8080/lk_infections";
let INFECTIONS = 'http://ec2-3-122-224-7.eu-central-1.compute.amazonaws.com:8080/lk_infections?days=0&factor=1';
// @TODO: work with the following url in the slide function, if the api providing the data.
// let INFECTIONS_DAY_PARAM = 'http://ec2-3-122-224-7.eu-central-1.compute.amazonaws.com:8080/lk_infections?days=_DAY_&factor=1';

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

	callAPI('GET', INFECTIONS, input,
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
	points2 = [];
	var input = {};

	let INFECTIONS_DAY_PARAM;

	// @TODO: work with the following url in the slide function, if the api providing the data.
	// INFECTIONS_DAY_PARAM = INFECTIONS_DAY_PARAM.replace('_DAY_', day);
	INFECTIONS_DAY_PARAM = INFECTIONS;
	callAPI('GET', INFECTIONS_DAY_PARAM, input,
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
		});
}

function heatmapFunctionAbsolute() {
	points2 = [];

	var input = {};

	callAPI('GET', INFECTIONS, input,
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
var ratio = 200.0; // I don't like the ration / why is this value so large needed?

function cleanHeatmap() {
	map.removeLayer(heatmap);
	heatmap = null;
}

function showHeatmap(heatdata, maxValue) {
	if (heatmap === null) {
		heatmap = L.heatLayer(heatdata, 
		{radius: 35, max: maxValue / ratio,
		blur: 25,
		gradient: {0.0: 'blue', 0.5: 'lime', 1: 'red'}
		}).addTo(map);
	} else {
		cleanHeatmap();
	}
	
	return heatmap;
}
