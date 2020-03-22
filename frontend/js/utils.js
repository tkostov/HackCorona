date = new Date();
document.getElementById("date").innerHTML = date.getDate() + "." + (date.getMonth() + 1) + "." + date.getFullYear();;

var slider = document.getElementById("rangeDays");
var additionalDays = document.getElementById("additionalDays");
var socialDistancing = document.getElementById("socialDistancing");
additionalDays.innerHTML = slider.value - 1;

slider.onchange = function() {
    app.refreshMap(slider.value - 1, socialDistancing.value === 'on' ? 1 : 0);
    additionalDays.innerHTML = this.value - 1;
};

socialDistancing.onchange = function () {
    app.refreshMap(slider.value - 1, socialDistancing.value === 'on' ? 1 : 0);
};