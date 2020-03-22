let app = new Vue({
  el: '.container-fluid',
  data: {
    name: 'Vue.js'
  },
  // define methods under the `methods` object
  methods: {
    heatmapFunctionPercentage: function (event) {
      heatmapFunctionPercentage();
    },
    heatmapFunctionAbsolute: function (event) {
      heatmapFunctionAbsolute();
    },
    refreshMap: function (day, socialDist) {
      refreshMap(day, socialDist);
    }
  }
});