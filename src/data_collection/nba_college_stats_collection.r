#Load package
library(cbbdata)

#Login
cbd_login()

#Choose the player
player_name <- "Stephen Curry"

#Fetch season stats for every player
seasons <- cbd_torvik_player_season(player_name)

#Export to CSV
write.csv(seasons, "college_seasons.csv", row.names = FALSE)

print("Export complete")