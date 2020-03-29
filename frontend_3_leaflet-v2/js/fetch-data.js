function fetchAndProcessData() {
    $.ajax({
        url: 'http://ec2-3-122-224-7.eu-central-1.compute.amazonaws.com:8080/ch_infections',
        type: 'GET',

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
    console.log(data);
}