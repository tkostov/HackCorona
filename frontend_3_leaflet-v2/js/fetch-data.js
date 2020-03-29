function fetchAndProcessData() {
    $.ajax({
        url: 'http://ec2-3-122-224-7.eu-central-1.compute.amazonaws.com:8080/ch_infections',
        type: 'GET',
        dataType: 'json',
        data: {
            format: 'json'
        },
        error: function () {
            console.error('an error occurred');
        },
        success: function (data) {
            processData(data);
        }
    });
}

function processData(data) {
    _apiData = data.rows;
    $(data.rows).each(function(i, obj) {
        rowData = {
            'density': obj[0],
            'lat': obj[1],
            'lng': obj[2],
            'datetime': obj[3],
            'day': obj[3].substr(0, 10),
        };
        if (!_apiData.hasOwnProperty('day-'+rowData.day)) {
            _apiData['day-'+rowData.day] = [];
            _apiData['day-'+rowData.day].push(rowData);
        } else {
            _apiData['day-'+rowData.day].push(rowData);
        }
    });
    drawPointLayer();

}