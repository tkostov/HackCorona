let app = new Vue({
  el: '.container-fluid',
  data: {
    name: 'Vue.js'
  },
  // define methods under the `methods` object
  methods: {
    heatmapFunctionCases: function (event) {
      console.log('Triiger A');
      heatmapFunctionCases();
    },
    heatmapFunctionDeads: function (event) {
      console.log('Triiger B');
      heatmapFunctionDeads();
    }
  }
})