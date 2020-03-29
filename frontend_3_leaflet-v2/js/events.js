$(document).ready(function() {
    fetchAndProcessData();
    $('#play').click(function(){play();});
    //var _daysSlider = document.getElementById("rangeDays");
    //_daysSlider.onchange = function() {handleDaysSlider(this); };
});

function handleDaysSlider(slider)
{
    var day = slider.value;
    $('#selectedDay').html(day);
    
}


function play() 
{
    $('#daysPanel p').each(function(i, obj) {
       var handle = this; 
       setTimeout(function() {showDataOfDay(handle)}, 1000);
       //sleep(1000);
    });
}