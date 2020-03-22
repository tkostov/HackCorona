date = new Date();
document.getElementById("date").innerHTML = date.getDate() + "." + (date.getMonth() + 1) + "." + date.getFullYear();;

var slider = document.getElementById("rangeDays");
var additionalDays = document.getElementById("additionalDays");
additionalDays.innerHTML = slider.value - 1;

slider.oninput = function() {
    app.slide(this.value - 1);
    additionalDays.innerHTML = this.value - 1;
}	

