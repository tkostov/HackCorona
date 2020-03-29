let app = new Vue({
  el: '.container-fluid',
  data: {
    name: 'Vue.js'
  },
  // define methods under the `methods` object
  methods: {
    heatmapFunctionPercentage: function () {
      type = 'percentage';
      refreshMap(day, socialDist);

      if ( document.getElementById("btnPercentage").classList.contains('btn-secondary') && !document.getElementById("btnPercentage").classList.contains('btn-dark'))
          document.getElementById("btnPercentage").classList.toggle('btn-dark');
      if ( document.getElementById("btnAbsolute").classList.contains('btn-dark') )
          document.getElementById("btnAbsolute").classList.toggle('btn-dark');
    },
    heatmapFunctionAbsolute: function () {
      type = 'absolute';
      refreshMap(day, socialDist);

      if ( document.getElementById("btnAbsolute").classList.contains('btn-secondary') && !document.getElementById("btnAbsolute").classList.contains('btn-dark') )
          document.getElementById("btnAbsolute").classList.toggle('btn-dark');
      if ( document.getElementById("btnPercentage").classList.contains('btn-dark') )
          document.getElementById("btnPercentage").classList.toggle('btn-dark');
    },
    refreshMap: function (day, socialDist) {
      refreshMap(day, socialDist);
    }
  }
});