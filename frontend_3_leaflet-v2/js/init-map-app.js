(function strechMapAfterInit() {
  let map = document.getElementById("map");
  map.style.height = "100%";
  map.style.width = "100%";
})();

(function loadMap(id, zoomLevel) {
  map = L.map('map').setView([46.208588, 6.145976], zoomLevel);
  L.tileLayer('https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token=pk.eyJ1IjoibWFwYm94IiwiYSI6ImNpejY4NXVycTA2emYycXBndHRqcmZ3N3gifQ.rJcFIG214AriISLbB6B5aw',
    {
      maxZoom: 15,
      minZoom: 5,
      attribution: 'Map data &copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, ' +
        '<a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, ' +
        'Imagery © <a href="https://www.mapbox.com/">Mapbox</a>',
      id: 'mapbox.light'
    }).addTo(map);

  // Old way to show the demo data
  // addressPoints = addressPoints.map(function (p) {
  //   return [p[0], p[1]];
  // });

  //var heat = L.heatLayer([]).addTo(map);
})('map', 7);

document.getElementById("btnAbsolute").classList.toggle('btn-dark');
//refreshMap(document.getElementById("rangeDays").value, document.getElementById("socialDistancing").checked ? 1 : 0);

function drawPointLayer() 
{
    var dataOfTheDay = _apiData['day-2020-03-27'];
    $(dataOfTheDay).each(function(i, obj) {
        L.circle([obj.lat, obj.lng], 5000).addTo(map);
        
    });
    
}