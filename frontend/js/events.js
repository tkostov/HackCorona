let app = new Vue({
  el: '.container-fluid',
  data: {
    name: 'Vue.js'
  },
  // define methods under the `methods` object
  methods: {
    heatmapFunctionPercentage: function (event) {
      cleanHeatmap();
      heatmapFunctionPercentage();
    },
    heatmapFunctionAbsolute: function (event) {
      cleanHeatmap();
      heatmapFunctionAbsolute();
    },
    slide: function (day) {
      cleanHeatmap();
      slide(day);
    }
  }
})