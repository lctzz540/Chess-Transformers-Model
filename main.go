package main

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"sync"

	"github.com/notnil/chess"
)

func processMatchHistory(history string) ([]string, string) {
	moves := []string{}
	result := ""

	movesData := strings.Fields(history)
	for _, move := range movesData {
		if strings.Contains(move, ".") {
			continue // Skip move numbers
		}

		move = strings.Trim(move, ".!?") // Remove punctuation marks

		if result == "" && (move == "1-0" || move == "0-1" || move == "1/2-1/2") {
			result = move
		} else {
			moves = append(moves, move)
		}
	}

	return moves, result
}

func processMatch(game *chess.Game, writer *csv.Writer, count *int, wg *sync.WaitGroup) {

	for game.Outcome() == chess.NoOutcome {
		moves := game.ValidMoves()
		move := moves[rand.Intn(len(moves))]
		game.Move(move)
	}

	moves, result := processMatchHistory(game.String())

	if result == "1-0" {
		writer.Write(moves)
		(*count)++
	}
}

func main() {
	file, err := os.OpenFile("moves_history.csv", os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0644)
	if err != nil {
		fmt.Println("Error opening CSV file:", err)
		return
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	var wg sync.WaitGroup
	targetMatches := 1000
	threadCount := 200
	count := 0
	countMutex := sync.Mutex{}

	for i := 0; i < threadCount; i++ {
		wg.Add(1)

		go func() {
			for {
				countMutex.Lock()
				if count >= targetMatches {
					countMutex.Unlock()
					break
				}
				countMutex.Unlock()

				game := chess.NewGame()
				processMatch(game, writer, &count, &wg)
			}
		}()
	}

	wg.Wait()

	fmt.Println("Moves history written to moves_history.csv")
}
