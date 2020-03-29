<?php

$minLatitude = 51.000000000000000;
$storedLatitude = [];
$maxLatitude = 52.000000000000000;

$minLongitude = 8.000000000000000;
$storedLongitude = [];
$maxLongitude = 9.000000000000000;

function generateUniqueNumber($minFloat, $maxFloat, array &$storage) {
  $gotUniqueFloat = FALSE;
  while ($gotUniqueFloat !== TRUE) {
    $randomFloat = float_rand($minFloat, $maxFloat);
    if (!in_array($randomFloat, $storage, TRUE)) {
      $storage[] = $randomFloat;
      $gotUniqueFloat = TRUE;
    }
  }

  return $randomFloat;
}

$finalNumbersArray = [];

for ($i = 0; $i <= 10000; ++$i) {
  $uniqueLatitude = generateUniqueNumber($minLatitude, $maxLatitude, $storedLatitude);
  $uniqueLongitude = generateUniqueNumber($minLongitude, $maxLongitude, $storedLongitude);

  $finalNumbersArray[] = [$uniqueLatitude, $uniqueLongitude, 0];
}


file_put_contents(__DIR__ . '/../data/points.js', 'var addressPoints =');
file_put_contents(__DIR__ . '/../data/points.js', json_encode($finalNumbersArray), FILE_APPEND);
file_put_contents(__DIR__ . '/../data/points.js', ';', FILE_APPEND);

/**
 * Generate Float Random Number
 *
 * @param float $Min Minimal value
 * @param float $Max Maximal value
 * @param int $round The optional number of decimal digits to round to. default 0 means not round
 * @return float Random float value
 */
function float_rand($Min, $Max, $round=0){
  //validate input
  if ($Min>$Max) { $min=$Max; $max=$Min; }
  else { $min=$Min; $max=$Max; }
  $randomfloat = $min + mt_rand() / mt_getrandmax() * ($max - $min);
  if($round>0)
    $randomfloat = round($randomfloat,$round);

  return $randomfloat;
}